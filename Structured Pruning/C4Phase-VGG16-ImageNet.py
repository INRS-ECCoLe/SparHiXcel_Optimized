import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import random
import os
import assign_PE_max_output_filter_cpp
import torchvision.models as models


# =================================================================
# 2. CORE PRUNER: Row-Balanced & Hardware-Surgical Logic
# =================================================================
class CustomIterativePruner:
    def __init__(self, model, conv_max, fc_max, phase_threshold=0.25, p3_start=0.7, arg1=33, arg2=60, max_out_filt=256, max_mux_trans=10):
        self.model = model
        self.conv_max = conv_max  
        self.fc_max = fc_max      
        self.phase_threshold = phase_threshold
        self.p3_start = p3_start
        self.arg1 = arg1 
        self.arg2 = arg2 
        self.max_out_filt = max_out_filt
        self.max_mux_trans = max_mux_trans
        self.masks = {}
        self.scores = {}
        self.pruneload = {}
        self._init_structures()

    def _init_structures(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.masks[name] = torch.ones_like(module.weight.data)
                self.scores[name] = torch.zeros_like(module.weight.data)
                if isinstance(module, nn.Conv2d):
                    self.pruneload[name] = 0.0

    def get_sparsity(self, name):
        mask = self.masks[name]
        return 1.0 - (mask.sum().item() / mask.numel())

    def apply_masks(self):
        for name, module in self.model.named_modules():
            if name in self.masks:
                m = self.masks[name].to(module.weight.device)
                module.weight.data *= m
                if module.weight.grad is not None:
                    module.weight.grad *= m

    def get_phase3_targets(self, layer_name, top_group_pct):
        module = dict(self.model.named_modules())[layer_name]
        w_np = np.transpose(module.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
        snaps, assigns, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(self.arg1, self.arg2, self.max_out_filt, w_np, self.max_mux_trans)
        
        W_H = module.kernel_size[0]
        ch_per_slice = self.arg1 // W_H
        groups = {}
        for i, (snap, assign) in enumerate(zip(snaps, assigns)):
            if not assign['filters']: continue # Skip empty PEs
            gid = (assign['filters'][0] // self.max_out_filt, assign['channels'][0] // ch_per_slice)
            if gid not in groups: groups[gid] = []
            waste = sum([self.arg2 - row_val for row_val in snap])
            groups[gid].append({'snap': snap, 'assign': assign, 'waste': waste, 'pe_idx': i})
        
        g_stats = [{'gid': k, 'avg_waste': sum(p['waste'] for p in v)/len(v), 'pes': v, 'pe_count': len(v)} for k, v in groups.items()]
        g_stats.sort(key=lambda x: x['avg_waste'], reverse=True)
        return g_stats[:max(1, int(len(g_stats) * top_group_pct))]

    def find_best_candidate_for_group(self, layer_name, group_pes, limit):
        module = dict(self.model.named_modules())[layer_name]
        w_data, W_H = module.weight.data, module.kernel_size[0]
        curr_zeros = (self.masks[layer_name] == 0).sum().item()
        total_elems = self.masks[layer_name].numel()
        
        pe_options = []
        for pe in group_pes:
            snap = pe['snap']
            max_val = max(snap)
            if max_val == 0: continue
            bottleneck_channels = [r // W_H for r in range(0, len(snap), W_H) if snap[r] == max_val]
            
            pe_indices, pe_mag = [], 0.0
            for ch_idx in bottleneck_channels:
                if ch_idx >= len(pe['assign']['channels']): continue
                phys_ch = pe['assign']['channels'][ch_idx]
                best_k_mag, best_k_indices = float('inf'), []
                for phys_f in pe['assign']['filters']:
                    mask_k = self.masks[layer_name][phys_f, phys_ch]
                    p_counts = (mask_k == 0).sum(dim=1)
                    c_min = p_counts.min().item()
                    target_p = c_min + 1
                    if target_p > module.kernel_size[1]: continue
                    
                    k_mag, k_indices = 0.0, []
                    for row in range(W_H):
                        needed = int(max(0, target_p - p_counts[row].item()))
                        if needed > 0:
                            row_w = w_data[phys_f, phys_ch, row, :]
                            active = (mask_k[row, :] == 1)
                            if active.sum() < needed: k_mag = float('inf'); break
                            vals, idxs = torch.topk(torch.abs(row_w[active]), k=needed, largest=False)
                            k_mag += vals.sum().item()
                            row_active_indices = active.nonzero(as_tuple=True)[0]
                            for sub_idx in idxs: k_indices.append((phys_f, phys_ch, row, row_active_indices[sub_idx].item()))
                    if k_mag < best_k_mag: best_k_mag, best_k_indices = k_mag, k_indices
                if best_k_indices: pe_mag += best_k_mag; pe_indices.extend(best_k_indices)
            
            # Precision Guard: Only allow if it stays under max_conv limit
            if pe_indices:
                projected_sparsity = (curr_zeros + len(pe_indices)) / total_elems
                if projected_sparsity <= limit:
                    pe_options.append({'indices': pe_indices, 'mag': pe_mag})
        
        return min(pe_options, key=lambda x: x['mag']) if pe_options else None

    def prune_layer(self, name, module, step_pct):
        curr_s = self.get_sparsity(name)
        target_max = self.conv_max.get(name, 0.0) if isinstance(module, nn.Conv2d) else self.fc_max.get(name, 0.0)
        if curr_s >= target_max: return True 
        actual_step = min(step_pct, target_max - curr_s)
        num_to_prune = int(module.weight.numel() * actual_step)
        if num_to_prune <= 0: return True
        if isinstance(module, nn.Conv2d): self._prune_conv(name, module, num_to_prune, curr_s)
        else: self._prune_fc(name, module, num_to_prune)
        return self.get_sparsity(name) >= target_max

    def _prune_conv(self, name, module, num_to_prune, curr_s):
        mask, w = self.masks[name], module.weight.data
        for _ in range(num_to_prune):
            p_row = (mask == 0).sum(dim=3)
            valid = (p_row <= p_row.min(dim=2, keepdim=True)[0]).unsqueeze(-1).expand_as(mask)
            candidates = (mask == 1) & valid
            if curr_s < self.phase_threshold:
                idx = self._select_random_min_mag(w, candidates)
            else:
                max_s = self.scores[name][candidates].max()
                idx = self._select_random_min_mag(w, candidates & (self.scores[name] == max_s))
            if not idx: break
            mask[idx], w[idx] = 0, 0
            self._update_scores(name, idx[0], idx[1], module.weight.shape[2])

    def _select_random_min_mag(self, w, candidates):
        indices = candidates.nonzero()
        if len(indices) == 0: return None
        thresh = torch.quantile(torch.abs(w[candidates]), 0.1)
        small = indices[torch.abs(w[indices[:,0], indices[:,1], indices[:,2], indices[:,3]]) <= thresh]
        return tuple(random.choice(small).tolist())

    def _update_scores(self, name, o, i, kh):
        counts = (self.masks[name][o, i] == 0).sum(dim=1)
        for h in range(kh): self.scores[name][o, i, h, :] = (counts.max() - counts[h]).float()

    def _prune_fc(self, name, module, num_to_prune):
        mask, w = self.masks[name], module.weight.data
        for _ in range(num_to_prune):
            cands = (mask == 1).nonzero()
            if len(cands) == 0: break
            thresh = torch.quantile(torch.abs(w[mask == 1]), 0.1)
            small = cands[torch.abs(w[cands[:,0], cands[:,1]]) <= thresh]
            idx = tuple(random.choice(small).tolist())
            mask[idx], w[idx] = 0, 0

# =================================================================
# 3. UTILITIES & REPORTING
# =================================================================
def get_loaders(batch_size=256):
    # Standard Nibi/Compute Canada ImageNet paths (Verify these on your server)
    traindir = "/datashare/imagenet/ILSVRC2012/train"
    valdir = "/datashare/imagenet/ILSVRC2012/val"
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # num_workers should match the number of CPU cores requested in your SLURM script
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=16, pin_memory=True)
    
    return train_loader, val_loader

def get_accuracy(model, loader):
    model.eval()
    t1, t5, total = 0, 0, 0
    dev = next(model.parameters()).device
    with torch.no_grad():
        for img, target in loader:
            img, target = img.to(dev), target.to(dev)
            out = model(img)
            _, pred = out.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            t1 += correct[:1].reshape(-1).float().sum(0).item()
            t5 += correct[:5].reshape(-1).float().sum(0).item()
            total += target.size(0)
    return (t1/total)*100, (t5/total)*100

def train_epoch(model, loader, opt, crit, pruner=None):
    model.train()
    dev = next(model.parameters()).device
    for img, target in loader:
        img, target = img.to(dev), target.to(dev)
        opt.zero_grad()
        crit(model(img), target).backward()
        
        # 1. Zero out gradients for pruned weights before the step
        if pruner: pruner.apply_masks()
        
        # 2. Optimizer step (This applies gradients, but secretly adds momentum)
        opt.step()
        
        # 3. IRONCLAD GUARD: Re-apply masks immediately after the step
        # This permanently forces all dead weights to exactly 0.0, killing the ghost weights.
        if pruner: pruner.apply_masks()

def print_layer_sparsity(pruner):
    print("\n========== Per-layer Sparsity Report ==========")
    tz, tp = 0, 0
    for name, mask in pruner.masks.items():
        z, p = (mask == 0).sum().item(), mask.numel()
        tz += z; tp += p
        print(f"{name:40s} | Sparsity: {100.*z/p:6.2f}%")
    print(f"------------------------------------------------")
    print(f"GLOBAL SPARSITY: {100.*tz/tp:6.2f}%")
    print("===============================================\n")

def run_utilization_analysis(model):
    print("\n========== Hardware Utilization Analysis (C++) ==========")
    arg1, arg2, arg3, max_mux = 33, 60, 256, 10
    for name, module in model.named_modules():
        # Check for both standard and quantized Conv2d
        if isinstance(module, (nn.Conv2d, torch.nn.intrinsic.qat.ConvBnReLU2d, torch.nn.intrinsic.ConvReLU2d, torch.nn.quantized.Conv2d)):
            try:
                # Handle standard vs quantized weight retrieval
                if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                    w = module.weight
                elif hasattr(module, 'weight') and callable(module.weight):
                    w = module.weight()
                else:
                    continue

                # Move to CPU and transpose for the C++ mapper
                w_np = np.transpose(w.dequantize().detach().cpu().numpy() if w.is_quantized else w.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
                
                # Call C++ Wrapper
                snaps, assigns, fit = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(arg1, arg2, arg3, w_np, max_mux)
                print(f"Layer: {name:25s} | Utilization: {fit:6.2f}% | PEs Used: {len(snaps)}")
            except Exception as e:
                print(f"Layer: {name:25s} | Skipping utilization check due to error: {e}")

def report_sparsity_from_quantized_model(model):
    print("\nSparsity from quantized (post-QAT) weights:")
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
            w = module.weight().dequantize()
            print(f"{name:30s} | Sparsity: {100.*(w==0).sum().item()/w.numel():6.2f}%")

def report_uniform_row_sparsity_from_quantized_model(model):
    print("\nUniform row-wise sparsity from quantized (post-QAT) weights:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Conv2d):
            w = module.weight().dequantize()
            uz = sum((w[o, i] == 0).sum(dim=1).min().item() * w.shape[2] for o in range(w.shape[0]) for i in range(w.shape[1]))
            print(f"{name:30s} | QuantizedConv2d | Uniform row sparsity: {100.*uz/w.numel():6.2f}%")

# =================================================================
# 4. SURGICAL LOOP & QAT
# =================================================================
def run_phase3_surgical_loop(model, pruner, config):
    target_groups = {}
    top_group_pct = config.get('top_group_pct', 0.1)
    
    # --- NEW: ACCUMULATION FAST-FORWARD ---
    ready_to_prune = False
    attempts = 0
    
    while not ready_to_prune:
        attempts += 1
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                curr_s = pruner.get_sparsity(name)
                limit = config['max_conv'].get(name, 0.0)
                if curr_s >= limit:
                    continue

                # 1. Increment Pruneload
                pruner.pruneload[name] += top_group_pct

                # 2. Calculate Threshold
                num_filters = module.out_channels
                size_penalty = max(1.0, 256.0 / num_filters) if num_filters < 256 else 1.0
                
                # We need unique groups to find the cost per group
                w_np = np.transpose(module.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
                snaps, assigns, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(
                    pruner.arg1, pruner.arg2, pruner.max_out_filt, w_np, pruner.max_mux_trans
                )
                module = model.get_submodule(name)
                W_H = module.kernel_size[0]
                unique_gids = set()
                groups_data = {}
                for i, (snap, assign) in enumerate(zip(snaps, assigns)):
                    if not assign['filters']: continue
                    gid = (assign['filters'][0] // pruner.max_out_filt, assign['channels'][0] // (pruner.arg1 // W_H))
                    unique_gids.add(gid)
                    if gid not in groups_data: groups_data[gid] = []
                    waste = sum([pruner.arg2 - row_val for row_val in snap])
                    groups_data[gid].append({'snap': snap, 'assign': assign, 'waste': waste, 'pe_idx': i})

                total_groups_count = len(unique_gids)
                if total_groups_count == 0: continue
                
                threshold = (1.0 / total_groups_count) * size_penalty

                # 3. Check if ready
                if pruner.pruneload[name] >= threshold:
                    ready_to_prune = True
                    num_groups_to_target = int(pruner.pruneload[name] // threshold)
                    
                    # Deduct cost
                    pruner.pruneload[name] -= (num_groups_to_target * threshold)

                    # Prepare targets
                    g_stats = [{'gid': k, 'avg_waste': sum(p['waste'] for p in v)/len(v), 'pes': v, 'pe_count': len(v)} 
                               for k, v in groups_data.items()]
                    g_stats.sort(key=lambda x: x['avg_waste'], reverse=True)
                    
                    targets = g_stats[:num_groups_to_target]
                    for g in targets: g['original_pe_count'] = g['pe_count']
                    target_groups[name] = targets
                    
                    print(f"   [Phase 3] Iteration {attempts}: {name} ready! Targeting {num_groups_to_target} groups.")
                else:
                    target_groups[name] = []
        
        # Guard: If everything is finished or we are stuck, break to avoid infinite loop
        if attempts % 100 == 0:
            print(f"   [Phase 3] Still accumulating load... (Attempt {attempts})")
        if attempts > 1000: 
            print("   [Phase 3] Reached 1000 accumulation attempts without a target. Stopping.")
            return False

    any_actual_prune_total = False
    
    while any(target_groups.values()):
        groups_to_remove = [] # Track groups that are stuck
        
        for name in list(target_groups.keys()):
            layer_limit = config['max_conv'].get(name, 0.0)
            remaining_groups_in_layer = []
            
            for g in target_groups[name]:
                cand = pruner.find_best_candidate_for_group(name, g['pes'], layer_limit)
                if cand:
                    any_actual_prune_total = True
                    target_layer = model.get_submodule(name)
                    for idx in cand['indices']:
                        pruner.masks[name][idx] = 0
                        target_layer.weight.data[idx] = 0
                    remaining_groups_in_layer.append(g)
                else:
                    # If this group is stuck (limit hit), don't add to remaining
                    print(f"   [Phase 3] Group in {name} is stuck at limit. Removing from targets.")
            
            target_groups[name] = remaining_groups_in_layer

        # If no groups are left in the whole model, stop
        if not any(target_groups.values()):
            break

        # --- Hardware Feedback Loop (C++) ---
        for name in list(target_groups.keys()):
            if not target_groups[name]: continue
            module = dict(model.named_modules())[name]
            w_np = np.transpose(module.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
            snaps, assigns, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(
                pruner.arg1, pruner.arg2, pruner.max_out_filt, w_np, pruner.max_mux_trans
            )
            
            W_H = module.kernel_size[0]
            new_counts = {}
            for snap, assign in zip(snaps, assigns):
                if not assign['filters']: continue 
                gid = (assign['filters'][0] // pruner.max_out_filt, assign['channels'][0] // (pruner.arg1 // W_H))
                new_counts[gid] = new_counts.get(gid, 0) + 1
            
            before = len(target_groups[name])
            # Keep only groups that HAVEN'T reduced their PE count yet
            target_groups[name] = [g for g in target_groups[name] if new_counts.get(g['gid'], 0) >= g['original_pe_count']]
            
            if len(target_groups[name]) < before:
                print(f"   [Phase 3] {before - len(target_groups[name])} groups in {name} optimized (PE count reduced).")

            # Update snapshots for the groups that are still working
            for g in target_groups[name]:
                g['pes'] = [{'snap': snaps[i], 'assign': assigns[i]} for i, a in enumerate(assigns) 
                            if a['filters'] and (a['filters'][0] // pruner.max_out_filt, a['channels'][0] // (pruner.arg1 // W_H)) == g['gid']]
    
    return any_actual_prune_total
# --- PHASE 4 UTILITIES ---

def get_pe_mapping_state(model, pruner):
    state = {}
    pe_counts = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            W_H = module.kernel_size[0]
            w_np = np.transpose(module.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
            snaps, assigns, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(
                pruner.arg1, pruner.arg2, pruner.max_out_filt, w_np, pruner.max_mux_trans)
            
            pe_counts[name] = len(snaps) # THE ABSOLUTE PHYSICAL TRUTH
            
            groups = {}
            for assign in assigns:
                if not assign['filters']: continue
                gid = (assign['filters'][0]//pruner.max_out_filt, assign['channels'][0]//(pruner.arg1//W_H))
                if gid not in groups: groups[gid] = []
                groups[gid].append((tuple(sorted(assign['filters'])), tuple(sorted(assign['channels']))))
            for gid in groups: groups[gid] = sorted(groups[gid])
            state[name] = groups
    return state, pe_counts

def run_phase4_step1_revival(model, pruner, original_weights):
    """
    STEP 1: Deterministic Kernel Row Balancing.
    Fills rows to match the density of the most dense row in each kernel.
    """
    print("\n--- Phase 4: Step 1 (Kernel Row Balancing) ---")
    revived_count = 0
    for name, module in model.named_modules():
        if name in pruner.masks and isinstance(module, nn.Conv2d):
            mask = pruner.masks[name]
            orig_w = original_weights[f"{name}.weight"].to(mask.device)
            kh, kw = module.kernel_size
            
            # Iterate through every individual kernel (Out_ch, In_ch)
            for o in range(mask.shape[0]):
                for i in range(mask.shape[1]):
                    kernel_mask = mask[o, i] # (KH, KW)
                    pruned_per_row = (kernel_mask == 0).sum(dim=1)
                    min_pruned = pruned_per_row.min().item()
                    
                    for r in range(kh):
                        needed = int(pruned_per_row[r].item() - min_pruned)
                        if needed > 0:
                            # Identify candidates in this row (currently pruned)
                            dead_indices = (kernel_mask[r] == 0).nonzero(as_tuple=True)[0]
                            # Select by largest magnitude in original weights
                            row_orig_mags = torch.abs(orig_w[o, i, r, dead_indices])
                            _, top_sub_idxs = torch.topk(row_orig_mags, k=needed)
                            
                            for sub_idx in top_sub_idxs:
                                actual_idx = dead_indices[sub_idx]
                                mask[o, i, r, actual_idx] = 1
                                model.get_submodule(name).weight.data[o, i, r, actual_idx] = orig_w[o, i, r, actual_idx]
                                revived_count += 1
    print(f"   Step 1 complete. Revived {revived_count} elements.")

def get_layer_hardware_state(layer_name, model, pruner, W_H_map):
    """Helper to quickly grab the current C++ assignment state of a single layer."""
    module = dict(model.named_modules())[layer_name]
    w_np = np.transpose(module.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
    _, assigns, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(
        pruner.arg1, pruner.arg2, pruner.max_out_filt, w_np, pruner.max_mux_trans)
    
    W_H = W_H_map[layer_name]
    groups = {}
    for assign in assigns:
        if not assign['filters']: continue
        gid = (assign['filters'][0]//pruner.max_out_filt, assign['channels'][0]//(pruner.arg1//W_H))
        if gid not in groups: groups[gid] = []
        groups[gid].append((tuple(sorted(assign['filters'])), tuple(sorted(assign['channels']))))
    for gid in groups: groups[gid] = sorted(groups[gid])
    return groups

def check_pe_counts(model, pruner, tag=""):
    """Helper to do a completely independent, raw count of all PEs in the model."""
    print(f"\n   [CHECKPOINT: {tag}] Auditing Physical Hardware...")
    pe_counts = {}
    total_pes = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w_np = np.transpose(module.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
            snaps, _, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(
                pruner.arg1, pruner.arg2, pruner.max_out_filt, w_np, pruner.max_mux_trans)
            #print(snaps)
            pe_counts[name] = len(snaps)
            total_pes += len(snaps)
            print(f"      -> {name}: {len(snaps)} PEs")
    return pe_counts, total_pes
def sanitize_hardware_ghosts(model, pruner):
    print("\n--- Pre-Phase 4 Hardware Sanitization ---")
    total_ghosts_killed = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in pruner.masks:
            w = module.weight.data
            mask = pruner.masks[name]
            
            # 1. Exact C++ scale calculation
            max_val = w.max().item()
            min_val = w.min().item()
            abs_max = max(abs(max_val), abs(min_val))
            
            if abs_max == 0:
                continue # Skip if layer is entirely 0
                
            scale = (2.0 * abs_max) / 255.0
            near_zero_threshold = 2.0 ** -8
            
            # 2. Simulate C++ Quantization exactly
            quantized_w = torch.round(w / scale)
            
            # 3. Identify Ghost Weights (PyTorch says alive, C++ says dead)
            # C++ considers it dead if quantized val is 0 OR absolute val <= near_zero_threshold
            cpp_dead = (torch.abs(quantized_w) == 0) | (torch.abs(w) <= near_zero_threshold)
            ghosts = (mask == 1) & cpp_dead
            
            ghost_count = ghosts.sum().item()
            
            if ghost_count > 0:
                print(f"   -> {name}: Found {ghost_count} ghost weights. Forcing to 0.0")
                
                # 4. Permanently execute the ghost weights in both structures
                mask[ghosts] = 0
                w[ghosts] = 0.0
                total_ghosts_killed += ghost_count
                
    print(f"Sanitization complete. {total_ghosts_killed} ghost weights permanently removed.\n")

def run_phase4_step2_surgical_revival(model, pruner, config, original_weights):
    print("\n--- Phase 4: Step 2 (Iterative Surgical Back-filling) ---")
    
    # Crush any PyTorch momentum ghost weights before we start
    pruner.apply_masks() 
    
    # Dynamically pull hardware constraints directly from the synced pruner
    W_H_map = {name: m.kernel_size[0] for name, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    ARG1, ARG2, ARG3, MAX_MUX = pruner.arg1, pruner.arg2, pruner.max_out_filt, pruner.max_mux_trans
    
    print("Capturing global hardware baseline fingerprints...")
    baseline_state, baseline_pe_counts = get_pe_mapping_state(model, pruner) 
    
    group_trackers = []
    for layer_name, groups in baseline_state.items():
        for gid in groups.keys():
            group_trackers.append({
                'layer': layer_name, 'gid': gid, 'current_pe_ptr': 0, 
                'current_ch_ptr': 0, 'failed_candidates': set(), 'finished': False
            })

    total_revived, round_num, total_groups = 0, 1, len(group_trackers)

    while any(not g['finished'] for g in group_trackers):
        round_candidates, revived_this_round = [], 0 
        
        # --- PROPOSAL PHASE ---
        for g in group_trackers:
            if g['finished']: continue
            layer_name, gid = g['layer'], g['gid']
            module = model.get_submodule(layer_name)
            orig_w = original_weights[f"{layer_name}.weight"].to(pruner.masks[layer_name].device)
            W_H = W_H_map[layer_name]
            w_np = np.transpose(module.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
            snaps, assigns, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(ARG1, ARG2, ARG3, w_np, MAX_MUX)
            
            relevant_pes = []
            for snap, assign in zip(snaps, assigns):
                if not assign['filters']: continue
                curr_gid = (assign['filters'][0]//ARG3, assign['channels'][0]//(ARG1//W_H))
                if curr_gid == gid: relevant_pes.append({'snap': snap, 'assign': assign})

            if g['current_pe_ptr'] >= len(relevant_pes):
                g['finished'] = True
                continue

            pe = relevant_pes[g['current_pe_ptr']]
            num_slots = len(pe['snap']) // W_H
            
            found_candidate_in_round = False
            for step in range(num_slots):
                ch_idx = (g['current_ch_ptr'] + step) % num_slots
                snap_idx = ch_idx * W_H
                if snap_idx >= len(pe['snap']) or pe['snap'][snap_idx] == 0 or pe['snap'][snap_idx] >= ARG2 or ch_idx >= len(pe['assign']['channels']): continue
                
                phys_ch = pe['assign']['channels'][ch_idx]
                best_k_score, best_k_indices = -1.0, None
                for phys_f in pe['assign']['filters']:
                    if (phys_f, phys_ch) in g['failed_candidates']: continue
                    mask_k = pruner.masks[layer_name][phys_f, phys_ch]
                    if (mask_k == 0).sum(dim=1).min() > 0:
                        score, indices = 0.0, []
                        for r in range(W_H):
                            dead_row_indices = (mask_k[r] == 0).nonzero(as_tuple=True)[0]
                            mags = torch.abs(orig_w[phys_f, phys_ch, r, dead_row_indices])
                            val, sub_idx = torch.max(mags, dim=0)
                            score += val.item()
                            indices.append((phys_f, phys_ch, r, dead_row_indices[sub_idx].item()))
                        if score > best_k_score: best_k_score, best_k_indices = score, indices
                
                if best_k_indices:
                    g['proposal'] = {'indices': best_k_indices, 'f_c': (best_k_indices[0][0], best_k_indices[0][1]), 'target_ch': ch_idx, 'pe_capacity': num_slots}
                    round_candidates.append({'group': g, 'indices': best_k_indices})
                    found_candidate_in_round = True
                    break
            
            if not found_candidate_in_round:
                g['current_pe_ptr'] += 1
                g['current_ch_ptr'] = 0
                g['failed_candidates'] = set()
                if g['current_pe_ptr'] >= len(relevant_pes): g['finished'] = True

        if not round_candidates: break 

        # --- EXECUTION PHASE ---
        for cand in round_candidates:
            lyr = cand['group']['layer']
            target_layer = model.get_submodule(lyr)
            for idx in cand['indices']:
                pruner.masks[lyr][idx] = 1
                target_layer.weight.data[idx] = original_weights[f"{lyr}.weight"][idx]

        # --- VERIFICATION PHASE ---
        verified_layers = set(c['group']['layer'] for c in round_candidates)
        for layer_name in verified_layers:
            module = model.get_submodule(layer_name)
            w_np = np.transpose(module.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
            snaps, new_assigns, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(ARG1, ARG2, ARG3, w_np, MAX_MUX)
            
            # The Ghost Killer
            physical_leak = len(snaps) > baseline_pe_counts[layer_name]
            
            W_H = W_H_map[layer_name]
            new_layer_groups = {}
            for assign in new_assigns:
                if not assign['filters']: continue
                gid = (assign['filters'][0]//ARG3, assign['channels'][0]//(ARG1//W_H))
                if gid not in new_layer_groups: new_layer_groups[gid] = []
                new_layer_groups[gid].append((tuple(sorted(assign['filters'])), tuple(sorted(assign['channels']))))

            for gid in new_layer_groups: new_layer_groups[gid] = sorted(new_layer_groups[gid])

            mismatched_gids = set()
            for gid, baseline_arrays in baseline_state[layer_name].items():
                if new_layer_groups.get(gid, []) != baseline_arrays: mismatched_gids.add(gid)
            for gid in new_layer_groups:
                if gid not in baseline_state[layer_name]: mismatched_gids.add(gid)

            layer_round = [c for c in round_candidates if c['group']['layer'] == layer_name]
            active_gids = set(c['group']['gid'] for c in layer_round)
            innocent_leaks = mismatched_gids - active_gids

            target_layer = model.get_submodule(layer_name)

            if physical_leak and not mismatched_gids:
                innocent_leaks.add("GHOST_PE")

            if innocent_leaks:
                for cand in layer_round:
                    g = cand['group']
                    for idx in cand['indices']:
                        pruner.masks[layer_name][idx] = 0
                        target_layer.weight.data[idx] = 0
                    g['failed_candidates'].add(g['proposal']['f_c'])
            else:
                for cand in layer_round:
                    g = cand['group']
                    if g['gid'] in mismatched_gids or physical_leak:
                        for idx in cand['indices']:
                            pruner.masks[layer_name][idx] = 0
                            target_layer.weight.data[idx] = 0
                        g['failed_candidates'].add(g['proposal']['f_c'])
                    else:
                        cap = g['proposal']['pe_capacity']
                        g['current_ch_ptr'] = (g['current_ch_ptr'] + 1) % cap
                        g['failed_candidates'] = set() 
                        weights_restored = len(cand['indices'])
                        total_revived += weights_restored
                        revived_this_round += weights_restored
        
        active_groups = sum(1 for g in group_trackers if not g['finished'])
        
        # --- REPORTING PHASE ---
        # 1. Standard Progress Summary (Every 50 rounds)
        if round_num % 200 == 0:
            print(f"   -> Round {round_num:05d} | Active Groups: {active_groups:03d}/{total_groups:03d} | Revived Now: {revived_this_round:04d} | Total Revived: {total_revived:05d}")
        
        # 2. Detailed Hardware PE Array Report (Every 100 rounds)
        if round_num % 500 == 0:
            print(f"\n   === HARDWARE PE ARRAY REPORT (Round {round_num:05d}) ===")
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv2d):
                    w_np = np.transpose(mod.weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float64)
                    snaps, _, _ = assign_PE_max_output_filter_cpp.assign_PE_max_output_filter(ARG1, ARG2, ARG3, w_np, MAX_MUX)
                    print(f"      -> Layer: {name:20s} | PEs Used: {len(snaps)}")
            print("   =================================================\n")

        round_num += 1

    print(f"\n   [✔] Step 2 surgical revival complete. Final revived count: {total_revived} weights.")

def apply_qat_and_save(model, pruner, loaders, save_path):
    print("\nStarting QAT with Frozen Weights...")
    model.cpu().train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(model, inplace=True)
    model.cuda()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100): 
        train_epoch(model, loaders[0], opt, nn.CrossEntropyLoss(), pruner)
        t1, top5 = get_accuracy(model, loaders[1])
        model.train()
        print(f"QAT Epoch {epoch+1} Top1: {t1:.2f}%, Top5: {top5:.2f}%")
        # After epoch 2, freeze BN to stabilize
        if epoch > 2:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    model.cpu().eval()
    torch.ao.quantization.convert(model, inplace=True)
    
    report_uniform_row_sparsity_from_quantized_model(model)
    report_sparsity_from_quantized_model(model)
    run_utilization_analysis(model)
    torch.save(model.state_dict(), save_path)

# =================================================================
# 5. HYBRID PRUNING ENGINE
# =================================================================
def run_hybrid_pruning(model, loaders, config):
    train_ld, val_ld = loaders
    pruner = CustomIterativePruner(
    model, 
    config['max_conv'], 
    config['max_fc'], 
    config['p1_threshold'], 
    config['p3_start'], 
    max_out_filt=config.get('max_out_filt', 256),
    max_mux_trans=config.get('max_mux_trans', 10)
    )
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    ref_t1, ref_t5 = get_accuracy(model, val_ld)
    # This line prints it to the terminal
    print(f"Base Accuracy - Top1: {ref_t1:.2f}%, Top5: {ref_t5:.2f}%")
   
    while True:
        conv_masks = [m for k, m in pruner.masks.items() if 'conv' in k or 'features' in k]
        cnn_s = sum((m == 0).sum().item() for m in conv_masks) / sum(m.numel() for m in conv_masks)
        checkpoint = (
            {k: v.cpu() for k, v in model.state_dict().items()}, 
            {k: v.cpu() for k, v in pruner.masks.items()}, 
            {k: v.cpu() for k, v in pruner.scores.items()}
        )
         # Determine which drop threshold to use
        is_p3 = cnn_s >= (config['p3_start'] - 0.001)
        current_max_drop = config['max_drop_p3'] if is_p3 else config['max_drop']
        if not is_p3:
            step = min(config['step_pct'], config['p3_start'] - cnn_s)
            any_active = False
            for name, module in model.named_modules():
                if name in pruner.masks:
                    if not pruner.prune_layer(name, module, step): any_active = True
            
            print(f"\n[Phase 1/2] Sparsity: {cnn_s:.4f} | Target Drop: {current_max_drop}%")
            run_utilization_analysis(model)
            print_layer_sparsity(pruner)

            if not any_active: break
        else:
            print(f"--- Entering Phase 3 Surgical Mode (CNN Sparsity: {cnn_s:.2f}) ---")
            if not run_phase3_surgical_loop(model, pruner, config): break
            # Update sparsity after P3
            conv_masks = [m for k, m in pruner.masks.items() if 'conv' in k or 'features' in k]
            cnn_s = sum((m == 0).sum().item() for m in conv_masks) / sum(m.numel() for m in conv_masks)
            run_utilization_analysis(model)
            print_layer_sparsity(pruner)

        failed = False
        epochs = config['p3_epochs'] if is_p3 else config['max_epochs']
        for e in range(epochs):
            train_epoch(model, train_ld, opt, nn.CrossEntropyLoss(), pruner)
            scheduler.step()
            cur_t1, cur_top5 = get_accuracy(model, val_ld)
            print(f"  Recovery Epoch {e+1}: Top1: {cur_t1:.2f}%, Top5: {cur_top5:.2f}%, (CNN Sparsity: {cnn_s:.4f})")
            if (ref_t5 - cur_top5) <= current_max_drop: break
        else: failed = True

        if failed:
            print("\n[!] Accuracy drop too high. Rolling back.")
            #model.load_state_dict(checkpoint[0])
            #pruner.masks, pruner.scores = checkpoint[1], checkpoint[2]; break 
            # 1. Load weights and immediately move them back to the GPU
            # .to(device) ensures the model stays on the GPU for the next attempt
            model.load_state_dict({k: v.to(device) for k, v in checkpoint[0].items()})
    
            # 2. Restore masks and scores to the GPU
            pruner.masks = {k: v.to(device) for k, v in checkpoint[1].items()}
            pruner.scores = {k: v.to(device) for k, v in checkpoint[2].items()}
            # NEW: Clear the cache now that we've restored the clean state
            torch.cuda.empty_cache()
    
            # 3. Apply masks immediately to fix the weights
            pruner.apply_masks()
    
            break
    return model, pruner

# =================================================================
# 6. MAIN
# =================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    loaders = get_loaders(batch_size=256)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    t1, top5 = get_accuracy(model, loaders[1])
    print(f"Pre-pruning: Top1: {t1:.2f}%, Top5: {top5:.2f}%")
    # 0. SAVE ORIGINAL WEIGHTS FOR REVIVAL
    original_weights = copy.deepcopy(model.state_dict())
    max_conv_config = {
    name: 0.9 for name, m in model.named_modules() 
    if isinstance(m, nn.Conv2d)
    }
    max_conv_config['features.0'] = 0.5
    max_conv_config['features.2'] = 0.8
    config = {
        'max_conv': max_conv_config, 'max_fc': {name: 0.0 for name, m in model.named_modules() if isinstance(m, nn.Linear)}, 'step_pct': 0.05,
        'max_drop': 2.0,'max_drop_p3': 3.5, 'p1_threshold': 0.25, 'p3_start': 0.7, 'p3_epochs': 50,  
        'max_epochs': 100, 'top_group_pct': 0.3, 'max_out_filt': 256,
        'max_mux_trans': 10
    }
    model, pruner = run_hybrid_pruning(model, loaders, config)
    t1, top5 = get_accuracy(model, loaders[1])
    print(f"Final Reports Before REVIVAL STEP 1 : Top1: {t1:.2f}%, Top5: {top5:.2f}%")
    run_utilization_analysis(model)
    print_layer_sparsity(pruner)
    
    # 3. PHASE 4: REVIVAL STEP 1
    run_phase4_step1_revival(model, pruner, original_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4) 
    # Lower LR (0.001) for revival to protect the established living weights
    # Retrain for Step 1
    print("\nRetraining after Step 1 Revival...")
    for epoch in range(30): # User-defined epochs
        train_epoch(model, loaders[0], optimizer, nn.CrossEntropyLoss(), pruner)
        t1, top5 = get_accuracy(model, loaders[1])
        print(f"Phase 4-1 Epoch {epoch+1}: Top1: {t1:.2f}%, Top5: {top5:.2f}%")
    # Report Step 1
    run_utilization_analysis(model)
    print_layer_sparsity(pruner)

    sanitize_hardware_ghosts(model, pruner)
    print_layer_sparsity(pruner)
    # 4. PHASE 4: REVIVAL STEP 2
    run_phase4_step2_surgical_revival(model, pruner, config, original_weights)
    sanitize_hardware_ghosts(model, pruner)
    print_layer_sparsity(pruner)
    # Retrain for Step 2
    print("\nRetraining after Step 2 Revival...")
    for epoch in range(50): # User-defined epochs
        train_epoch(model, loaders[0], optimizer, nn.CrossEntropyLoss(), pruner)
        t1, top5 = get_accuracy(model, loaders[1])
        print(f"Phase 4-2 Epoch {epoch+1}: Top1: {t1:.2f}%, Top5: {top5:.2f}%")
    # Final Reports before Quantization
    run_utilization_analysis(model)
    print_layer_sparsity(pruner)
    t1, top5 = get_accuracy(model, loaders[1])
    print(f"Final Reports before Quantization {epoch+1}: Top1: {t1:.2f}%, Top5: {top5:.2f}%")
    # 5. QAT & SAVE
    apply_qat_and_save(model, pruner, loaders, "final_quant_revived.pth")
    print(f"""
    ================================================================================
    FINAL EXPERIMENT SUMMARY
    ================================================================================
    * sanity bug for ghost weights after revival phase is fixed.
    * Learning Rate Policy: Increased LR in QAT from 0.0001 to 0.001.
    * Pruning Policy: Phase 3 max accuracy drop set to 3.5% (Evaluating Top-5).
    * Batch sized increased from 128 to 256.
    * high Phase 2 to 70% and lower phase 1 to 25%.
    * higher batch size to 256.
    *'top_group_pct': 0.3
    * max_conv_config['features.0'] = 0.5.
    * max_conv_config['features.2'] = 0.8.
    * Total Retraining (Revivals):epochs increased .
    * Total Retraining (Quantization): epochs increased.
    ================================================================================
    """)