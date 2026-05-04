import numpy as np
import assign_PE_max_output_filter_cpp as assign_PE_max_output_filter
import pretrained_models_torchvision
from pretrained_models_torchvision import get_pruned_weights_dict
from tqdm import tqdm
import time
import os
import csv
# =====================================================================
# --- MAIN EXECUTION ---
# =====================================================================

start_time = time.time()
log_filename = "efficientnet_all_layers12x48-00.txt"
log_filename_csv = "efficientnet_all_layers12x48-00.csv"

# Optional: Clear the log file at the start of a new run
header = ["Layer_Name", "kH", "kW", "C_in", "C_out", "Fitness", "Count"]
with open(log_filename_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
with open(log_filename, "w") as f:
    f.write(f"Evaluation Run Started: {time.ctime()}\n\n")

# --- Load pruned weights ---
print("Loading and pruning weights...")
weights_dict = get_pruned_weights_dict(pruning_amount=0.70)

# --- Constants for fitness function ---
arg1, arg2, arg3, max_mux_trans = 12, 48, 256, 10  

# Filter for 4D convolution layers only
conv_layers = {name: weights for name, weights in weights_dict.items() if len(weights.shape) == 4}
print(f"\nFound {len(conv_layers)} convolutional layers. Starting evaluation...\n")

# --- Execution Loop ---
for layer_name, layer_weights in tqdm(conv_layers.items(), desc="Evaluating Layers"):
    
    # --- Extract weights and transpose to [kH, kW, C_in, C_out] ---
    weights_np = np.transpose(layer_weights, (2, 3, 1, 0))
    kH, kW, C_in, C_out = weights_np.shape
    
    # Call C++ function
    fitness, count = assign_PE_max_output_filter.assign_PE_max_output_filter(
        arg1, arg2, arg3, weights_np, max_mux_trans
    )
    
    # Save to logs inside loop to mirror your GA structure
    with open(log_filename, "a") as f:
        f.write(f"Layer: {layer_name:<35} | Fitness: {fitness:.6f} | Count: {count}\n")
        f.flush() # This ensures it saves even if the system crashes
        
    with open(log_filename_csv, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([layer_name, kH, kW, C_in, C_out, f"{fitness:.6f}", count])

# --- Final output ---
end_time = time.time()

with open(log_filename, "a") as f:
    f.write(f"\n Finished at: {time.ctime()}\n")
    f.write(f"Total Runtime: {end_time - start_time:.2f} seconds\n")

print("\nStart:", time.ctime(start_time))
print("End:", time.ctime(end_time))
print("Total runtime:", end_time - start_time, "seconds")