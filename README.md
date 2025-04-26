# Diffusion-based Adversarial Purification over Latent Embeddings

This repository contains the implementation of our CS607 course project, **"Diffusion-based Adversarial Purification over Latent Embeddings"** — a novel method for adversarial defense that leverages diffusion models in the **latent space**.  
We use a **Pix2Pix-based encoder-decoder** architecture to project images into a compact latent space, perform **diffusion-based purification** to remove adversarial perturbations, and reconstruct clean images for robust classification.

Evaluated on the **ImageNet** dataset using a **ResNet-50** classifier under **PGD** and **FGSM** attacks ((\(\epsilon = 8/255, 16/255\))), our approach significantly boosts robust accuracy compared to unpurified adversarial images.

<br>

> **Authors**:  
> Bhavik Shangari (12240410), Uday Bhardwaj (12241910), Vedant Marodkar (12240990)  
> **Date**: April 27, 2025  
> **Course**: CS607 - Adversarial Machine Learning  
> **Repository**: [GitHub Link](https://github.com/BhavikShangari/AdversarialML_Project)

---

## 🚀 Project Overview

Adversarial attacks introduce small, often imperceptible perturbations to images, leading deep neural networks to make incorrect predictions. Traditional defenses like adversarial training are **attack-specific** and **computationally expensive**.

We propose an alternative: **adversarial purification** using **latent diffusion** — a process that removes adversarial noise before classification.  
Our method diffuses adversarial noise **directly over latent embeddings** (not raw images), preserving semantic content while being **computationally efficient**.

### 🔑 Key Contributions
- **Latent Diffusion**: Purification is done in a **512-dimensional latent space**, reducing computational overhead.
- **Pix2Pix Encoder-Decoder**: Skip connections ensure that semantic features are preserved during purification.
- **Robustness**: Achieves robust accuracies of:
  - **43.4%** on PGD attacks (\(\epsilon = 16/255\))  
  - **41.3%** on FGSM attacks (\(\epsilon = 16/255\))  
  (Compared to **4.7%** and **22.1%** respectively for unpurified adversarial images.)

---


## 🛠️ Pipeline Overview

![Purification Pipeline](Latent_DiffPure (1).png)


The purification pipeline consists of three major components:

- **Encoder**  
  Maps \(64\times64\times3\) images to **512-dimensional** latent embeddings using a convolutional network with LeakyReLU activations and batch normalization.

- **Diffusion Model**  
  Applies controlled noise to the latent space and denoises it using a feed-forward neural network conditioned on timesteps (DDPM-style scheduling).

- **Decoder**  
  Reconstructs purified \(64\times64\times3\) images from latent embeddings using a deconvolutional network with skip connections and ReLU activations.

> **Illustration**:  
> *Images → Latent Embeddings → Diffusion Purification → Reconstructed Images → Classification*

---

## 🧩 Repository Structure
```bash
.
├── create_adv_examples.ipynb      # Generate adversarial examples (PGD, FGSM)
├── DiffAE.ipynb                   # Train and evaluate purification pipeline
├── model_epoch_resnet50_epoch_30.pth # Pretrained ResNet-50 checkpoint
├── outputs/
│   ├── pipeline_checkpoints/      # Saved model checkpoints
│   ├── pipeline_plots/            # Plots during training (optional)
│   └── pipeline_samples/          # Sample images (training & validation)
├── README.md                      # This file
├── train_resnet.py                # Train ResNet-50 classifier
└── tree.txt                       # Repository overview
```

---

## 📦 Requirements

Make sure the following are installed:

- Python 3.8+
- PyTorch 1.9+
- torchvision
- NumPy
- Pillow
- Jupyter Notebook
- tqdm

Install them via:

```bash
pip install torch torchvision numpy pillow jupyter tqdm
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/BhavikShangari/AdversarialML_Project.git
cd AdversarialML_Project
```

### 2. Download ImageNet
- Download the ImageNet (ILSVRC2012) dataset from [image-net.org](https://image-net.org/).
- Resize images to \(64 \times 64\) and organize:
  ```
  ./data/imagenet/
  ├── train/  (100,000 images, 200 classes)
  └── val/    (10,000 images, 200 classes)
  ```

### 3. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install torch torchvision numpy pillow jupyter tqdm
```

---

## 🔥 Usage Instructions

### Step 1: Train ResNet-50
```bash
python train_resnet.py --data_dir ./data/imagenet --epochs 30 --batch_size 64 --lr 2e-4
```
- Outputs: `model_epoch_resnet50_epoch_30.pth`
- Adjust `--epochs`, `--batch_size`, or `--lr` as needed.

---

### Step 2: Generate Adversarial Examples

- Open the notebook:
```bash
jupyter notebook create_adv_examples.ipynb
```
- Configure:
  - Attack type: **PGD** or **FGSM**
  - Epsilon values: \((8/255), (16/255)\)
  - Checkpoint: `model_epoch_resnet50_epoch_30.pth`
- Run to generate adversarial examples for a 512-image subset.

---

### Step 3: Train and Evaluate the Purification Pipeline

- Open:
```bash
jupyter notebook DiffAE.ipynb
```
- Set parameters:
  - Dataset path: `./data/imagenet`
  - Epochs: `26`
  - Learning rate: `2e-4`
  - Diffusion timestep \(t\): 
    - \(t = 0.1\) for PGD
    - \(t = 0.075\) for FGSM

- Outputs are saved in `outputs/pipeline_samples/` every 10 epochs.

---

## 📊 Results

| Attack | Settings | Standard Acc | Adversarial Acc | Purified Acc |
|:------:|:--------:|:------------:|:---------------:|:------------:|
| PGD | \(\epsilon = 16/255, \alpha=4/255\) | 62.5% | 4.7% | **43.4%** |
| FGSM | \(\epsilon = 16/255\) | 62.5% | 22.1% | **41.3%** |

- **Raw Images**: 62.5% standard accuracy
- **Adversarial Images**: Accuracy drops to 4.7% (PGD) and 22.1% (FGSM)
- **Purified Images**: Accuracy restored to 43.4% (PGD) and 41.3% (FGSM)

---

## 📈 Reproducing Results

1. Train ResNet-50 via `train_resnet.py`.
2. Generate adversarial samples via `create_adv_examples.ipynb`.
3. Run the purification pipeline via `DiffAE.ipynb`.
4. Report accuracies following the same evaluation setup.

---

## 🙏 Acknowledgments

We thank the **CS607 course instructors** for their guidance throughout the project.  
Special thanks to the authors of **DiffPure** for their foundational work on diffusion-based adversarial purification.

This project was developed as part of the **Adversarial Machine Learning** course at **IIT Bhilai**.

---
