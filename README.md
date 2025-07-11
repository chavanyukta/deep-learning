Grapevine Leaf Species Classification Using CNN
Project Overview
This project implements a deep learning approach to classify grapevine leaf species using Convolutional Neural Networks (CNN). The model processes a dataset of annotated grapevine leaf images, leveraging data augmentation techniques like rotation, flipping, and scaling to improve robustness and generalization. This automated classification method aims to support agricultural practices by improving grapevine variety management, product quality, and sustainability.

Abstract
We developed a CNN model to classify different species of grapevine leaves from images. Data augmentation was applied to enhance model performance and generalization. Our model achieved an overall test accuracy of 89%, demonstrating high precision and recall in most categories. This work highlights the potential of CNNs in agricultural image classification tasks and lays groundwork for future improvements such as transfer learning and hybrid neural architectures.

Dataset
The dataset consists of labelled images of various grapevine leaf species.

Images are RGB and resized to a uniform size to fit the CNN input requirements.

Dataset split: 80% training, 20% testing.

Data augmentation applied during training: rotation, flipping, scaling.

Installation & Requirements
Python 3.8+

PyTorch

torchvision

PyTorch Lightning

numpy

pandas

You can install the required libraries using:

bash
Copy
pip install torch torchvision pytorch-lightning numpy pandas
Usage
Clone this repository:

bash
Copy
git clone https://github.com/chavanyukta/deeplearning.git
cd deeplearning
Run the Jupyter notebook Untitled11.ipynb to train and evaluate the CNN model.

Follow the notebook to see dataset preprocessing, model training, and results evaluation.

Model Architecture
Multiple convolutional layers with ReLU activation extract hierarchical features.

Max pooling layers reduce spatial dimensions and computational complexity.

Fully connected layers classify features into grapevine leaf species.

Output layer uses softmax activation to produce class probabilities.

Cross-Entropy loss and Adam optimizer used during training.

Data augmentation and dropout regularization applied to prevent overfitting.

Results
Test accuracy: 89%

Precision and recall scores range between 79% and 100% depending on species.

Confusion matrix and classification report provide detailed performance metrics.

The model performs exceptionally well in classes like 'Buzgulu' (100% precision).

Future Work
Explore transfer learning with pre-trained models to improve accuracy.

Increase dataset size, especially for underrepresented classes.

Use feature visualization (e.g., Grad-CAM) to interpret model decisions.

Implement continuous learning for model updates in production environments.

References
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 1097-1105.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition, 770-778.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. http://www.deeplearningbook.org

Author
Yukta Sanjiv Chavan
Student ID: A1873169
The University of Adelaide


