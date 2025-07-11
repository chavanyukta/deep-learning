# Grapevine Leaf Species Classification Using CNN

## Project Overview
This project implements a deep learning approach to classify grapevine leaf species using Convolutional Neural Networks (CNN). The model processes a dataset of annotated grapevine leaf images, leveraging data augmentation techniques like rotation, flipping, and scaling to improve robustness and generalization. This automated classification method aims to support agricultural practices by improving grapevine variety management, product quality, and sustainability.

## Abstract
We developed a CNN model to classify different species of grapevine leaves from images. Data augmentation was applied to enhance model performance and generalization. Our model achieved an overall test accuracy of 89%, demonstrating high precision and recall in most categories. This work highlights the potential of CNNs in agricultural image classification tasks and lays groundwork for future improvements such as transfer learning and hybrid neural architectures.

## Dataset
- The dataset consists of labelled images of various grapevine leaf species.
- Images are RGB and resized to a uniform size to fit the CNN input requirements.
- Dataset split: 80% training, 20% testing.
- Data augmentation applied during training: rotation, flipping, scaling.

## Installation & Requirements
- Python 3.8+
- PyTorch
- torchvision
- PyTorch Lightning
- numpy
- pandas
- 
## Project Structure
- Untitled11.ipynb # Jupyter notebook with model training and evaluation
- data/ # Directory containing grapevine leaf images
- README.md # Project documentation file
- requirements.txt # List of required Python libraries
- 
## Project Report
The full project report including methodology, results, and analysis can be found in the [`CNN Image Classification Report.pdf`](./CNN%20Image%20Classification%20Report.pdf) file.

## Contact
Yukta Sanjiv Chavan
Email: chavanyukta@gmail.com
