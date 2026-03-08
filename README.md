# FoldingNet From Scratch

---

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/status-research--project-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

A clean **from-scratch PyTorch implementation of** FoldingNet from the paper FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation.

The objective of this project is twofold:

1. **Reproduce FoldingNet training from scratch**
2. **Investigate alternative optimization methods**, specifically the Sparse Bregman Optimizer, and compare its performance with the AdamW optimizer used in the original paper.

---

### Navigation

[Overview](#project-motivation) •
[Architecture](#model-architecture) •
[Implementation](#implementation-details) •
[Training](#training) •
[Results](#example-results) •
[Issues](#known-issues-and-proposed-improvements) •
[Future Work](#future-work) •
[References](#references)

---

### Project Motivation

There are two motivational drivers that inform this project: academic and personal.

#### 1. Academic

3D point clouds are a fundamental representation used in areas such as:

* autonomous driving
* robotics
* augmented and virtual reality
* 3D reconstruction

However, point clouds are **unordered and irregular**, which makes them challenging for traditional neural network architectures. FoldingNet addresses this challenge by learning a **latent representation of a 3D shape** and reconstructing it by **deforming a 2D grid into the target geometry**.

Beyond the data modality and model architecture, optimization strategies are a very important aspect of deep learning. These strategies often work in tandem with some sort of objective function, which is usually augmented with regularization terms in a attempt to force the trained model into adopting desirable qualities. However, regularization terms are a passive means of ensuring desired properties i.e., parameter sparsity. A good question to ask is: can we enforce sparsity by altering, not the objective function, but the optimization strategy itself?

The Bregman framework is the answer to this question. This optimization framework allows neural networks to be trained from scratch, such that the optimization process only yields non-zero parameters when special mathematical circumstances necessitate this. This enforces sparsity as an internal measure of the optimization process itself.

#### 2. Personal

This project stands as a dive into the worlds of point clouds and 3D data science. It also serves to link this space  to the field of graph processing, which is of strong interest.

---

### Model Architecture

The implementation follows the architecture proposed in the original paper.

#### Encoder

The encoder extracts a **global shape descriptor** from the input point cloud.

Key components include:

* graph-based feature extraction
* local covariance features
* global max pooling for shape representation

#### Decoder (Folding-Based)

The decoder reconstructs the shape through **two folding operations**.

1. Start from a **2D grid**
2. Concatenate the grid with the latent vector
3. Apply two successive MLP folding layers
4. Deform the grid into the final **3D point cloud**

<img src="https://i-blog.csdnimg.cn/blog_migrate/898b9d776cf522b39053ad6edf9d0882.png" alt="FoldingNet architecture">
<p align="center"><strong>FoldingNet Architecture</strong></p>

---

### Implementation Details

The implementation follows the hyperparameters described in the original paper with **one modification**. The batch size was set to **16** instead of **1** due need for training speedups. All other architectural and training settings were kept consistent with the original work.

---

### Planned Research Extension

The original objective of this project was to evaluate the effect of **alternative optimization strategies** on the performance of the model. In particular, the project aims to integrate the Sparse Bregman Optimizer (a special optimization algorithm) for natively training sparse neural networks.

The goal of this experiment is to investigate:

* reconstruction quality improvements
* convergence behavior
* training stability compared to the optimizer used in the original paper

⚠️ The Sparse Bregman optimizer integration is **currently incomplete**, therefore the comparison experiments have not yet been conducted.

---

### Repository Structure

The repository tree is as below:

```text
.
├── LICENCE.md
├── README.md
├── notebook.ipynb
├── pyproject.toml
└── src
    ├── artefacts
    ├── data
    ├── eval.py
    ├── logs
    ├── main.py
    └── sparse_bregman
```

**NOTE**: The dataset is downloaded from Baidu and then decompressed into the data directory. As can be seen below, the data subdirectory has the following directory tree:

```text
src/data
└── shapenetpart_hdf5_2048
    ├── test0.h5
    ├── test0_id2file.json
    ├── test0_id2name.json
    ├── test1.h5
    ├── test1_id2file.json
    ├── test1_id2name.json
    ├── train0.h5
    ├── train0_id2file.json
    ├── train0_id2name.json
    ├── train1.h5
    ├── train1_id2file.json
    ├── train1_id2name.json
    ├── train2.h5
    ├── train2_id2file.json
    ├── train2_id2name.json
    ├── train3.h5
    ├── train3_id2file.json
    ├── train3_id2name.json
    ├── train4.h5
    ├── train4_id2file.json
    ├── train4_id2name.json
    ├── train5.h5
    ├── train5_id2file.json
    ├── train5_id2name.json
    ├── val0.h5
    ├── val0_id2file.json
    └── val0_id2name.json

83 directories, 30 files
```

---

### Installation

Just clone the repository, change directory into it and install dependencies via **uv**.

```bash
git clone https://dagshub.com/Harkhymadhe/sparse-bregman.git

cd sparse-bregman

uv pip install -e .
```

---

### Training

To train the model:

```bash
python3 src/main.py
```

Configuration parameters can be modified in the ```main.py``` script.

### Evaluation

To evaluate the trained model:

```bash
python3 src/eval.py
```

---

### Example Results

*(Visualization examples can be added here after training experiments.)*

Typical examples include:

* original point cloud
* reconstructed point cloud
* training loss curves

Example layout:

```
Original Shape        Reconstruction
     ◼                       ◼
   ◼ ◼ ◼                 ◼ ◼ ◼
  ◼     ◼               ◼     ◼
```

---

### Known Issues and Proposed Improvements

During development and experimentation with FoldingNet, several limitations and implementation challenges were identified. This section documents those issues along with potential improvements.

---

#### 1. Batch Size

**Issue**

The original work used a batch size of $1$, while this implementation uses a batch size of $16$. This results in a very different number of iterations total.

This likely may have influenced:

* gradient stability
* convergence speed
* final reconstruction performance

**Possible Fixes**

Potential improvements include:

* gradient accumulation to simulate larger batch sizes
* mixed precision training
* distributed training across multiple GPUs
* change learning rate to match p with ew batch size

---

#### 2. Incomplete Sparse Bregman Optimizer Integration

**Issue**

The comparison between the optimizer used in the original work and the Sparse Bregman Optimizer has not yet been completed. This is due to the fact that the Gregman optimization scheme has not been successfully implemented yet. As a result, the current implementation only supports the default optimizer used in the original FoldingNet paper (AdamW).

**Possible Fixes**

* Complete implementation of Sparse Bregman update rules.
* Integrate the optimizer into the PyTorch training pipeline
* Conduct controlled optimizer comparison experiments

---

#### 3. Limited Experimentation and Benchmarking

**Issue**

The current repository focuses mainly on **architectural reproduction**, and extensive benchmarking has not yet been conducted. As such, it is currently missing full training benchmarks, dataset comparisons and performance curves.

**Possible Fixes**

Future experiments should include:

* training on standard point cloud datasets
* comparison with results reported in the FoldingNet paper
* ablation studies on encoder and decoder components

---

#### 4. Visualization Tools

**Issue**

Point cloud reconstruction quality is best evaluated visually, but the repository currently lacks integrated visualization tools. In addition, it would make more visual sense to display the point cloud with some color information.

**Possible Fixes**

* Experiment with different tools and libraries for point cloud visualization (e.g., Open3D, PyVista).
* Estimate normal vector for all points in point cloud and use as proxy for color information.
* Since 2D grid is fixed and needs to deform to 3D space, try to assign color spectrum across the grid. Match colors to final 3D cloud.

---

#### 5. Reproducibility and Logging

**Issue**

Reproducibility is important for research projects, but presently experiment configuration management is currently limited. The hyperparameters are embedded directly in the ```main.py``` and ```eval.py``` files and this is not best practice. In addition, there are no real logs, beyond just a copy of the outputs emitted to the ternminal during training.

**Possible Fixes**

* Set seed for training.
* Decouple experiment hyperparameters and configuration from actual code. This can be done with:
  * Argparse
  * YAML
  * Dataclasses
* Configure logging. Possible tools for this include:
  * Loguru
  * Tensorboard
  * Weights & Biases (W&B)

---

#### 6. Objective Function

**Issue**

The objective function is an important cornerstone for the training process. The wrong function will likely lead to the wrong results. For this project, the objective function is a custom-designed form of the Chamfer's distance.

**Possible Fixes**

* Confirm that the extended Chamfer distance is implemented correctly.
* Experiment with different aggregation modes for the objective function (presently using sum).

---

#### 7. Grid Granularity

**Issue**

For the FoldingNet, the 2D grid must be deformed into 3D space to give the point cloud. The deformation (i.e., squeezing, cutting and folding) of this grid is guided by a  provided codeword, an embedding extracted from another point cloud, which represents an "idea" of what our 2D grid needs to be deformed into looking like.

First training experiments were done with the grid values between $-22.5$ and $22.5$. They were later done with values between $-3.0$ and $3.0$. It was visually observed that the second range yielded better outputs. It is hypothesized that this is because for a point cloud with coordinates in the range $-1.0$ to $1.0$, a 2D grid within the smaller range would have less deformation to undergo in order to conform to an appropriate 3D configuration.

**Possible Fixes**

* Do more extensive experimentation with grid granularity.
* Attempt to explore this hypothesis further.

---

#### 8. Point Cloud Size

**Issue**

Presently, using the **h5** ShapeNetPart dataset, the point clouds are restricted to $\le 2048$ points. This restricts the ability to experiment with higher fidelity point clouds.

**Possible Fixes**

This is already being fixed by:

* Obtaining ShapeNet part dataset with point clouds which contain more the $2048$ points.
* Implementing 3D point sampling on said dataset.

---

#### 9. Data Augmentation

**Issue**

Presently, in keeping with the  original FoldingNet paper, there is no augmentation applied to the point clouds. This would be a good idea to implement however, as it makes for a more robust model. Examples of augmentations that can be applied to the point clouds include:

1. Point subsampling (i.e., trying to artificially force the model to reconstruct meaningfully, even with hazy inputs).
2. Point jittering (i.e., adding random noise to the 3D coordinates of the points in the cloud; makes for model robustness).
3. Cloud rotation (i.e., reconstruction should be possible, irrespective of the cloud's orientation).

**Possible Fixes**

Implement data augmentation schemes as suggested and train again.

---

### References

1. [FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation](https://arxiv.org/abs/1712.07262)
2. [A Bregman Learning Framework for Sparse Neural Networks](https://jmlr.org/papers/v23/21-0545.html)
