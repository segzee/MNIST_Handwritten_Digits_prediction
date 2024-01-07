# Introduction
In this project, you will build a neural network to evaluate the MNIST dataset, a commonly used dataset in the field of machine learning. The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0 through 9) and is widely used for training and testing various machine learning models.

# Benchmark results on MNIST by notable researchers include:

88% accuracy: Lecun et al., 1998
95.3% accuracy: Lecun et al., 1998
99.65% accuracy: Ciresan et al., 2011
This project aims to develop a neural network model that performs well on the MNIST dataset, serving as a benchmark for your understanding of neural network architectures and training processes.

# Imports
The project utilizes essential Python libraries and PyTorch modules. Please refrain from changing the contents of the import cell as it includes necessary components for the project.

# Load the Dataset
The MNIST dataset is loaded using the torchvision module. It includes training and test sets, and the data is preprocessed using specified transformations. The dataset is split into training and validation sets, and data loaders are created for efficient batch processing during training.

# Justify your preprocessing
The preprocessing steps involve converting images to PyTorch tensors and normalizing pixel values. These steps enhance the efficiency of the neural network by providing standardized input data. ToTensor transformation converts images into tensors, and Normalize transformation centers pixel values around zero, making the model less sensitive to variations in pixel intensity.

# Explore the Dataset
Matplotlib, NumPy, and PyTorch are used to explore the dimensions of the data. A function show5 displays five images from a given data loader.

# Build your Neural Network
A simple neural network (SimpleNN) is defined using torch.nn layers. The architecture includes three fully connected layers with ReLU activation and dropout for regularization. The input size, hidden dimensions, and output size are adjustable parameters.

# Specify a loss function and an optimizer
The chosen loss function is CrossEntropyLoss, suitable for classification tasks. Stochastic Gradient Descent (SGD) is used as the optimizer, with a specified learning rate.

# Running your Neural Network
The neural network is trained using a specified number of epochs. Training and validation losses are recorded for each epoch. The model is moved to the GPU if available, and training progress is displayed.

Feel free to modify the code to experiment with different architectures, hyperparameters, or optimization techniques to achieve better performance on the MNIST dataset.
