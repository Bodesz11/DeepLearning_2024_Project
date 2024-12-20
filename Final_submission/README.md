# **Final Submission**

This repository contains the final submission for a deep learning-based image segmentation project. It includes all the necessary code, data, notebooks, and model evaluation processes required to train, evaluate, and visualize segmentation models.

## **Project Overview**

This project focuses on medical image segmentation using various deep learning architectures, including DeepLabV3, LinkNet, MANet, and U-Net. The repository is organized to facilitate data preprocessing, model training, evaluation, and visualization. The models have been evaluated using ensemble methods to enhance segmentation performance.

## **Repository Structure**

Final_submission/<br>
│<br>
├── **Reorganized_repo/**              # Directory containing organized project files<br>
│   ├── **DataLoader.py**              # Custom data loader for medical images and masks<br>
│   ├── **Data_preproc.py**            # Data preprocessing script (converts NIfTI to PNG, etc.)<br>
│   ├── **Evaluate_model.py**          # Functions for evaluating model performance<br>
│   ├── **Model.py**                   # Model definitions (DeepLabV3, LinkNet, MANet, U-Net)<br>
│   ├── **Plot_model_outputs.py**      # Visualization functions for model outputs and masks<br>
│   ├── **Train_model.py**             # Script for training models<br>
│   ├── **config.json**                # Configuration file (paths, model settings, hyperparameters)<br>
│   └── **main.py**                    # Main script to run the training and evaluation process<br>
│<br>
├── **data/**                          # Directory for storing data (images and masks)<br>
│<br>
├── **notebooks/**                     # Jupyter notebooks for exploration and visualization<br>
│   ├── **data_vis.ipynb**             # Data visualization and exploration notebook<br>
│   ├── **models_train.ipynb**         # Notebook for training the models<br>
│   ├── **models_test.ipynb**          # Notebook for testing the models<br>
│   └── **All_models_evaluation_and_thresholding.ipynb**  # Notebook for evaluating models and applying thresholds<br>
│<br>
├── **models/**                        # Directory to store trained models (not included in the tree) But the pre-trained models are availabe [here](https://drive.google.com/drive/folders/1HoITBBdEqvn5jGgTBSaabvxBBGlsXMt5?usp=sharing).<br>
│<br>
├── **README.md**                      # Main documentation for the repository<br>
│<br>
├── **requirements.txt**               # List of dependencies required for the project<br>
│<br>
└── **Final_submission/**              # Submission directory (contains project deliverables)<br>
    ├── **DeepLabV3.gv.png**           # Visualization of DeepLabV3 architecture<br>
    ├── **Linknet.gv.png**             # Visualization of LinkNet architecture<br>
    ├── **Manet.gv.png**               # Visualization of MANet architecture<br>
    ├── **U-net.gv.png**               # Visualization of U-Net architecture<br>
    └── **README.md**                  # Documentation specific to the submission<br>




## **How to Use**

1. **Setup the Environment**:
   - Install the required dependencies:
     ```
     pip install -r requirements.txt
     ```

2. **Preprocess Data**:
   - Use `Data_preproc.py` to convert raw NIfTI images to PNG format:
     ```
     python Data_preproc.py
     ```

3. **Train the Models**:
   - Run the main script to train models:
     ```
     python main.py
     ```

4. **Evaluate Models**:
   - Evaluate trained models using `Evaluate_model.py` or the Jupyter notebooks in `notebooks/`.

5. **Visualize Results**:
   - Use `Plot_model_outputs.py` for visualizing predictions and masks.

## **Dependencies**

- `torch`
- `torchvision`
- `segmentation_models_pytorch`
- `albumentations`
- `matplotlib`
- `numpy`
- `opencv-python`
- `pandas`
- `scikit-learn`
- `nibabel`
- `wandb`
