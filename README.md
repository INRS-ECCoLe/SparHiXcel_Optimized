# SparHiXcel-v2: Ordering Optimization & Structured Pruning

This repository contains the implementation of ordering optimization and structured pruning techniques developed as part of the research work:

**"High-Efficiency Sparsity-Aware FPGA Accelerator with Column-Wise Compression for Efficient CNN Inference"**

This work belongs to the **SparHiXcel-v2** framework and focuses on improving the efficiency of mapping sparse CNN models onto FPGA-based accelerator SparHiXcel.

---

## 📌 Overview

This repository provides:

- Genetic Algorithm (GA)-based filter/channel ordering optimization  
- Hardware-aware mapping under configurable constraints  
- Performance evaluation tools for FPGA deployment  
- Structured pruning using Surgical Iterative Pruning and Revival (SIPR)  

---
## 🛠️ Hardware Implementation

For the **HDL code** of the SparHiXcel accelerator, please visit the following repository:
👉 [SparHiXcel HDL (large-output-filter-based)](https://github.com/INRS-ECCoLe/SparHiXcel/tree/large-output-filter-based)

---


## 💻 Requirements & Dependencies

To run the codebase, you will need a system with **Python 3.11+**, a compatible **C++ compiler** (like GCC or MSVC), and a modern GPU for fast execution. 

### Python Libraries
Install the following libraries via `pip` or `conda`:
* `torch` (PyTorch)
* `torchvision`
* `numpy`
* `pybind11` (Crucial for building the C++ extension)
* `setuptools`

### Example Installation
```bash
# It is highly recommended to use a virtual environment
pip install torch torchvision numpy pybind11 setuptools
```

---
## 📂 Repository Structure
```text
.
├── ordering_optimization/
│   ├── Genetic_Algorithm.py
│   ├── performance.py
│   ├── pretrained_models_torchvision.py
│   ├── assign_PE_cpp.cpp
│   ├── setup.py
│   ├── ...
│
├── structured_pruning/
│   ├── SIPR_resnet18.py
│   ├── SIPR_vgg16.py
│   ├── setup.py
```

---

## ⚙️ Ordering Optimization

### 🔹 Model Selection

You can change the target model in `pretrained_models_torchvision.py` by modifying the following function:

```python
def get_pruned_weights_dict(...)
```

### 🔹 Genetic Algorithm (GA)

The GA implementation is available in `Genetic_Algorithm.py`. This allows you to run ordering optimization on a selected CNN layer with unstructured pruning.

**Hardware Constraints (Configurable):**
* `arg1` → Number of PE rows
* `arg2` → Number of PE columns
* `arg3` → Number of FSUM-Store blocks ($P$)
* `max_mux_trans` → MUX-T size ($T$)

**GA Hyperparameters:**
* `POP_SIZE`
* `NUM_GENERATIONS`
* `MUTATION_RATE`
* `TOURNAMENT_SIZE`
* `ELITISM_COUNT`

### 🔹 PE Assignment (C++ Backend)

The file `assign_PE_cpp.cpp` is responsible for Processing Element (PE) assignment. It must be compiled before use. It relies heavily on `pybind11` to interface with Python.

**Build Instructions:**
```bash
python setup.py build_ext --inplace
```

### 🔹 Performance Evaluation

Use `performance.py` to evaluate the mapping efficiency. This script allows you to:
1. Select a CNN layer
2. Define hardware constraints
3. Choose pruning levels
4. Evaluate mapping performance on the FPGA

---
## ✂️ Structured Pruning (SIPR)

Located in the `structured_pruning/` folder, this module includes implementations of Surgical Iterative Pruning and Revival (SIPR) for:
* **ResNet-18** → `C4Phase-Resnet18-ImageNet.py`
* **VGG-16** → `C4Phase-VGG16-ImageNet.py`

### 🔹 Setup Requirement
Before running structured pruning, you must build the C++ extension in this directory as well:
```bash
python setup.py build_ext --inplace
```

### 🔹 Training Details
* Initial dense pretrained models are obtained from Torchvision.
* Models are retrained multiple times during pruning.
* **Dataset:** ILSVRC2012 ImageNet
* **Batch size:** 256


---

## 🧠 Key Contributions

* Column-wise compression support.
* Hardware-aware ordering optimization of filter and channels using Genetic Algorithms.
* Structured pruning via Surgical Iterative Pruning and Revival (SIPR).

---

## 📄 Citation

If you use this repository, please cite:
> **High-Efficiency Sparsity-Aware FPGA Accelerator with Column-Wise Compression for Efficient CNN Inference**

*(A BibTeX format citation will be provided upon publication.)*

---

## ⚠️ Notes

* Ensure C++ extensions are compiled before running dependent scripts.


