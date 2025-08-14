# Perceptual Reality Transformer

[![arXiv](https://img.shields.io/badge/arXiv-2508.09852.svg)](https://arxiv.org/abs/2508.09852)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Official implementation for "Perceptual Reality Transformer: Neural Architectures for Simulating Neurological Perception Conditions"**

> **Abstract:** Neurological conditions affecting visual perception create profound experiential divides between affected individuals and their caregivers, families, and medical professionals. We present the Perceptual Reality Transformer, a comprehensive framework employing six distinct neural architectures to simulate eight neurological perception conditions with scientifically-grounded visual transformations. Our system learns mappings from natural images to condition-specific perceptual states, enabling others to experience approximations of simultanagnosia, prosopagnosia, ADHD attention deficits, visual agnosia, depression-related changes, anxiety tunnel vision, and Alzheimer's memory effects. Through systematic evaluation across ImageNet and CIFAR-10 datasets, we demonstrate that Vision Transformer architectures achieve optimal performance, outperforming traditional CNN and generative approaches. Our work establishes the first systematic benchmark for neurological perception simulation, contributes novel condition-specific perturbation functions grounded in clinical literature, and provides quantitative metrics for evaluating simulation fidelity. The framework has immediate applications in medical education, empathy training, and assistive technology development, while advancing our fundamental understanding of how neural networks can model atypical human perception.

---

## Overview

Neurological conditions affecting visual perception create profound experiential divides between affected individuals and their caregivers, families, and medical professionals. This repository presents the **Perceptual Reality Transformer**, a comprehensive framework employing six distinct neural architectures to simulate eight neurological perception conditions with scientifically-grounded visual transformations.

### Key Features

- **8 Neurological Conditions**: Simultanagnosia, prosopagnosia, ADHD attention, visual agnosia, depression, anxiety tunnel vision, Alzheimer's memory effects
- **6 Neural Architectures**: CNN, ResNet, Vision Transformer, LSTM, Diffusion, VAE
- **Comprehensive Evaluation**: 5 metrics across ImageNet and CIFAR-10 datasets
- **Clinically Grounded**: Perturbation functions derived from peer-reviewed neuroscience literature

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/linlab/prt.git
cd prt
pip install -r requirements.txt
```

### Basic Usage

```python
from models import ViTPerceptual
from perturbations import PERTURBATION_FUNCTIONS
import torch

# Load pretrained model
model = ViTPerceptual()
model.load_state_dict(torch.load('models/vit_best.pth'))

# Simulate a condition
image = torch.randn(1, 3, 224, 224)  # Your input image
condition = 1  # Simultanagnosia
severity = 0.8  # 80% severity

# Generate simulation
simulated = model(image, torch.tensor([condition]), torch.tensor([severity]))
```

### Command Line Interface

```bash
# Train all models on CIFAR-10
python main.py --models all --datasets cifar10 --epochs 50

# Train specific models
python main.py --models recurrent vit --datasets both --epochs 15
python main.py --models diffusion --datasets imagenet --epochs 15

# Fast test run
python main.py --models cnn --epochs 5 --samples-per-condition 100

# Parallel execution (run in separate terminals)
python main.py --models cnn residual --parallel-id 0
python main.py --models vit --parallel-id 1
```

## 📊 Results

Our comprehensive evaluation across CIFAR-10 and ImageNet demonstrates that **vision transformer** achieve optimal performance. A subset of the results are:

| Model | CIFAR-10 MSE ↓ | Diversity ↑ | Severity ↑ | ImageNet MSE ↓ |
|-------|----------------|-------------|------------|----------------|
| ViTPerceptual | **93,920** | 0.72 | **0.95** | **100,671** |
| EncoderDecoderCNN | 109,304 | 0.76 | 0.92 | 118,693 |

*Full results and analysis available in our [paper](https://arxiv.org/abs/2508.09852).*

## 🧠 Supported Conditions

| Condition | Description | Clinical Basis |
|-----------|-------------|----------------|
| **Simultanagnosia** | Cannot integrate multiple visual elements | Neitzel et al. (2016) |
| **Prosopagnosia** | Face recognition deficits | Eimer et al. (2012)|
| **ADHD Attention** | Visual attention disruption | Lin et al. (2017) |
| **Visual Agnosia** | Object recognition impairment | Clinical literature |
| **Depression** | Darkened, desaturated vision | Golomb et al. (2009) |
| **Anxiety** | Tunnel vision under stress | Dirkin (1983) |
| **Alzheimer's** | Progressive visual degradation | Rizzo et al. (2000) |

## 🏗️ Architecture Overview

```
Input Image → [Architecture Branch] → Condition Simulation
              ├── EncoderDecoderCNN: Basic encoder-decoder
              ├── ResidualNet: Residual perturbations  
              ├── ViTPerceptual: Vision transformer with attention
              ├── RecurrentLSTM: Sequential processing
              ├── DiffusionModel: DDPM-style generation
              └── GenerativeVAE: Latent space manipulation
```

## 📁 Repository Structure

```
prt/
├── models.py              # 7 neural architectures
├── perturbations.py       # Clinical perturbation functions  
├── data_utils.py          # Dataset loading and generation
├── training.py            # Training functions
├── evaluation.py          # Metrics and visualization
├── main.py               # Main execution script
├── cross_analysis.py     # Cross-dataset comparison
├── utils.py              # Utilities
└── outputs/              # Results, models, figures
    ├── CIFAR10/
    ├── ImageNet/
    └── cross_dataset_analysis/
```

## 📚 Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{lin2025perceptual,
  title={Perceptual Reality Transformer: Neural Architectures for Simulating Neurological Perception Conditions},
  author={Lin, Baihan},
  journal={arXiv preprint arXiv:2508.09852},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions from the community! Areas for contribution include:
- Adding new neurological conditions
- Implementing additional neural architectures
- Improving evaluation metrics
- Community validation studies with neurological condition groups

Please open an issue or pull request to discuss your ideas.

For questions about the paper or methodology, please refer to our [arXiv preprint](https://arxiv.org/abs/2508.09852), or contact me at: Baihan Lin (doerlbh@gmail.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
