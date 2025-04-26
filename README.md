Diffusion-based Adversarial Purification over Latent Embeddings

This repository contains the implementation of the CS607 project, "Diffusion-based Adversarial Purification over Latent Embeddings," a novel approach to adversarial purification using diffusion models in the latent dimension. The project leverages a Pix2Pix-based encoder-decoder architecture to map images to a compact latent space, purify adversarial perturbations via latent diffusion, and reconstruct clean images for classification. Evaluated on the ImageNet dataset with the ResNet-50 classifier against PGD and FGSM attacks ((\epsilon=8/255, 16/255)), our method significantly improves robust accuracy compared to unpurified adversarial images.
Authors: Bhavik Shangari (12240410), Uday Bhardwaj (12241910), Vedant Marodkar (12240990)Date: April 27, 2025Course: CS607 - Adversarial Machine LearningRepository: https://github.com/BhavikShangari/AdversarialML_Project
Project Overview
Adversarial attacks exploit vulnerabilities in deep neural networks by introducing imperceptible perturbations that lead to misclassifications. Traditional defenses, such as adversarial training, are computationally intensive and tailored to specific attack types. Adversarial purification offers a flexible alternative by using generative models to remove perturbations before classification. Our work introduces a novel purification pipeline that performs diffusion in the latent dimension, reducing computational complexity and preserving semantic information.
Key Contributions:

Latent Diffusion: Performs diffusion on 512-dimensional latent embeddings instead of high-dimensional images, enhancing computational efficiency.
Pix2Pix Architecture: Employs an encoder-decoder framework with skip connections to preserve semantic features during purification.
Robustness: Achieves robust accuracies of 43.4% (PGD, (\epsilon=16/255)) and 41.3% (FGSM, (\epsilon=16/255)) on ImageNet with ResNet-50, compared to 4.7% and 22.1% for unpurified adversarial images.

The pipeline is evaluated on the ImageNet dataset (100,000 training images, 10,000 validation images, 200 classes, resized to 64x64) using ResNet-50 against PGD ((\epsilon=8/255, 16/255), 20 iterations, step size (\epsilon/10)) and FGSM ((\epsilon=8/255, 16/255)) attacks.
Pipeline
The purification pipeline consists of three components:

Encoder: Maps 64x64x3 images to 512-dimensional latent embeddings using a convolutional neural network with LeakyReLU activations and batch normalization.
Diffusion Model: Applies controlled noise to latent embeddings and denoises them using a feed-forward neural network, conditioned on timesteps, with a DDPM scheduler.
Decoder: Reconstructs clean 64x64x3 images from purified embeddings using a deconvolutional network with skip connections and ReLU activations.

Figure 1: Adversarial purification pipeline. Images are encoded to a latent space, purified via diffusion, and decoded for classification.
Repository Structure
The repository is organized as follows:
.
├── create_adv_examples.ipynb       # Notebook to generate adversarial examples (PGD, FGSM)
├── DiffAE.ipynb                    # Main notebook for training and evaluating the purification pipeline
├── model_epoch_resnet50_epoch_30.pth  # Pre-trained ResNet-50 model checkpoint
├── outputs/
│   ├── pipeline_checkpoints/       # Model checkpoints for the purification pipeline
│   │   └── model_epoch_50.pth
│   ├── pipeline_plots/             # Plots generated during training (if any)
│   └── pipeline_samples/           # Sample images from training and validation (epochs 10, 20, 30, 40, 50)
├── README.md                       # This file
├── train_resnet.py                 # Script to train the ResNet-50 classifier
└── tree.txt                        # Repository structure

Requirements
To run the code, ensure the following dependencies are installed:

Python 3.8+
PyTorch 1.9+
torchvision
NumPy
Pillow
Jupyter Notebook
tqdm (for progress bars)

Install the dependencies using:
pip install torch torchvision numpy pillow jupyter tqdm

Setup Instructions

Clone the Repository:
git clone https://github.com/BhavikShangari/AdversarialML_Project.git
cd AdversarialML_Project


Download ImageNet:

Download the ImageNet dataset (ILSVRC2012) from image-net.org.
Preprocess the images to 64x64 resolution and organize them into training (100,000 images) and validation (10,000 images) sets across 200 classes.
Place the dataset in a directory, e.g., ./data/imagenet/.


Set Up the Environment:

Create a virtual environment (recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:pip install torch torchvision numpy pillow jupyter tqdm





Usage
1. Train the ResNet-50 Classifier
To train the ResNet-50 classifier on ImageNet:
python train_resnet.py --data_dir ./data/imagenet --epochs 30 --batch_size 64 --lr 2e-4


Outputs a pre-trained model checkpoint (model_epoch_resnet50_epoch_30.pth).
Adjust hyperparameters (e.g., --epochs, --batch_size, --lr) as needed.

2. Generate Adversarial Examples
Use the create_adv_examples.ipynb notebook to generate adversarial examples with PGD and FGSM attacks:

Open the notebook:jupyter notebook create_adv_examples.ipynb


Specify the attack parameters ((\epsilon=8/255, 16/255)) and the ResNet-50 checkpoint (model_epoch_resnet50_epoch_30.pth).
Run the cells to generate adversarial images for a test subset (512 images).

3. Train and Evaluate the Purification Pipeline
Use the DiffAE.ipynb notebook to train the purification pipeline and evaluate its performance:

Open the notebook:jupyter notebook DiffAE.ipynb


Configure the dataset path (./data/imagenet), epochs (26), learning rate (2e-4), and diffusion timesteps ((t^=0.1) for PGD, (t^=0.075) for FGSM).
Train the encoder, diffusion model, and decoder.
Evaluate on raw, adversarial, and purified images, measuring standard and robust accuracies.

Sample outputs (e.g., purified images) are saved in outputs/pipeline_samples/ for training and validation at epochs 10, 20, 30, 40, and 50.
Reproducing Results
To reproduce the results from the project report (Table 1):

Train ResNet-50 using train_resnet.py to achieve a standard accuracy of ~62.5%.
Generate adversarial examples using create_adv_examples.ipynb for PGD ((\epsilon=16/255, \alpha=4/255)) and FGSM ((\epsilon=16/255)).
Run the purification pipeline in DiffAE.ipynb to obtain robust accuracies (43.4% for PGD, 41.3% for FGSM).

Results
The purification pipeline significantly improves robust accuracy on ImageNet with ResNet-50:



Attack
Settings
Standard Acc
Adversarial Acc
Purified Acc



PGD
(\epsilon=16/255, \alpha=4/255)
62.5%
4.7%
43.4%


FGSM
(\epsilon=16/255)
62.5%
22.1%
41.3%



Raw Images: Standard accuracy of 62.5%.
Adversarial Images: PGD reduces accuracy to 4.7%, FGSM to 22.1% ((\epsilon=16/255)).
Purified Images: Purification restores accuracy to 43.4% (PGD) and 41.3% (FGSM).

Acknowledgments
We thank the CS607 course instructors for their guidance and the authors of DiffPure for their foundational work on diffusion-based purification. This project was developed as part of the Adversarial Machine Learning course.
Contact
For questions or issues, please open a GitHub issue or contact:

Bhavik Shangari: bhavik.shangari@example.com
Uday Bhardwaj: uday.bhardwaj@example.com
Vedant Marodkar: vedant.marodkar@example.com

