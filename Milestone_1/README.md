# Overview of Functions in first_model.ipynb
## 1. MedicalImageDataset Class
### Purpose: This class extends PyTorch's Dataset class to facilitate the loading and preprocessing of 3D medical images and their corresponding masks.
### Key Features:
#### Initialization: 
Takes in directories for images and masks, along with parameters for target size, transformations, and the number of segmentation classes.
#### Length Method:
Returns the total number of images in the dataset, allowing DataLoader to know how many samples are available.
#### Item Retrieval:
Loads and processes images and masks for a given index, including:
Loading NIfTI files using NiBabel.
Converting them to PyTorch tensors.
Resizing images and masks to a specified target size.
Applying optional transformations and clamping mask values to ensure they fall within the valid range of classes.
## 2. UNet Class
### Purpose: 
Implements the U-Net architecture, a powerful convolutional neural network commonly used for image segmentation tasks, particularly in medical imaging.
### Key Features:
### Architecture: 
Comprises an encoder-decoder structure with multiple convolutional blocks, max pooling layers for downsampling, and transposed convolution layers for upsampling.
### Forward Method: 
Defines how input data flows through the network, capturing both high-level features and spatial context through skip connections.
## 3. Data Preparation
### a. Dataset Creation
#### MedicalImageDataset: 
Initializes the dataset by specifying the directories containing the images and masks. It prepares the dataset for training and validation by splitting it into two parts—80% for training and 20% for validation.
### b. DataLoader
#### Purpose: 
Creates DataLoader instances for both the training and validation datasets. These DataLoader instances handle:
#### Batching: 
Groups data into batches for efficient processing.
#### Shuffling: 
Randomizes the order of data, which helps improve training robustness and model generalization.
## 4. Train Model Function
### Purpose: 
Orchestrates the training process of the U-Net model.
### Key Features:
#### Loss Tracking: 
Monitors both training and validation losses during the training process.
#### Optimization: 
Conducts forward and backward passes to update model weights.
#### Validation: 
Evaluates model performance on the validation set, saving the best model weights for later use based on validation loss.
#### Memory Management: 
Utilizes garbage collection to free up GPU memory, enhancing training efficiency.
## 5. Plotting the Results
### Functions:
#### plot_data and plot_mask: 
These functions enable visualization of the 3D medical images and their corresponding masks.
#### Batch Indexing: 
Allows specification of different mask layers or slices to be plotted, facilitating detailed analysis of the segmentation performance.
## 6. Model Testing
### Purpose: 
Loads the saved model and sets it to evaluation mode for testing.
### Process:
After running inference on the test dataset, it compares the original masks with the predicted masks.
Visual results are displayed, with differences highlighted, and saved as first_test.png for further inspection.
