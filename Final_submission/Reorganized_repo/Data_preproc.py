import os
import torch
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def list_files_with_suffix(directory, suffix):
    """List all files in a directory with a specific suffix."""
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)])


def generate_csv(data, output_path):
    """Generate a CSV file from a dictionary of data."""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"CSV saved to {output_path}")


def save_mask_as_png(mask_data, output_path, color_map):
    """
    Save a binary mask as a PNG image with a specific color.

    Parameters:
        mask_data (np.ndarray): The binary mask data (0s and 1s).
        output_path (str): Path where the PNG image will be saved.
        color_map (tuple): The RGB color to apply to the mask.
    """
    mask_img = np.zeros((mask_data.shape[0], mask_data.shape[1], 3), dtype=np.uint8)
    mask_img[mask_data == 1] = color_map
    img = Image.fromarray(mask_img)
    img.save(output_path)


def nii_to_png(nii_file_path, output_dir, axis=2, name='file'):
    """
    Convert a .nii.gz file to PNG images with separate masks for different classes.

    Parameters:
        nii_file_path (str): Path to the .nii.gz file.
        output_dir (str): Directory to save the PNG images.
        axis (int): Axis along which to extract slices (0, 1, or 2).
        name (str): Base name for the output files.
    """
    # Load the NIfTI file
    nii_img = nib.load(nii_file_path)
    data = nii_img.get_fdata()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Normalize data to 0-255 for PNG format
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)
    name = name.rstrip('.nii.gz.')

    # Loop through slices along the specified axis
    for i in range(data.shape[axis]):
        if axis == 0:
            slice_img = data[i, :, :]
        elif axis == 1:
            slice_img = data[:, i, :]
        else:
            slice_img = data[:, :, i]

        # Create three separate masks based on intensity values
        mask1 = (slice_img == 255).astype(np.uint8)  # White mask
        mask2 = (slice_img == 170).astype(np.uint8)  # Grey mask
        mask3 = (slice_img == 85).astype(np.uint8)   # Dark grey mask

        # Define a color for masks
        white_color = (255, 255, 255)

        # Save each mask as a PNG
        save_mask_as_png(mask1, os.path.join(output_dir, f"{name}_{i:03d}_mask1.png"), white_color)
        save_mask_as_png(mask2, os.path.join(output_dir, f"{name}_{i:03d}_mask2.png"), white_color)
        save_mask_as_png(mask3, os.path.join(output_dir, f"{name}_{i:03d}_mask3.png"), white_color)

    print(f"Masks saved to {output_dir}")


def preproc_main(input_dir, output_dir):
    """
    Converts all .nii.gz images and masks in the raw dataset to PNGs and generates CSV files listing them.

    Parameters:
        input_dir (str): The input directory containing raw .nii.gz images and masks.
        output_dir (str): The output directory where PNG images and CSV files will be saved.
    """
    # Define paths to NIfTI directories
    images_dir = os.path.join(input_dir, 'imagesTr')
    masks_dir = os.path.join(input_dir, 'labelsTr')

    # Convert images to PNG
    image_files = list_files_with_suffix(images_dir, '.nii.gz')
    for file in image_files:
        nii_to_png(os.path.join(images_dir, file), os.path.join(output_dir, 'imagesTr'), axis=2, name=file)

    # Convert masks to PNG
    mask_files = list_files_with_suffix(masks_dir, '.nii.gz')
    for file in mask_files:
        nii_to_png(os.path.join(masks_dir, file), os.path.join(output_dir, 'masksTr'), axis=2, name=file)

    # List PNG images and masks
    train_images = list_files_with_suffix(os.path.join(output_dir, 'imagesTr'), '.png')
    train_mask1 = list_files_with_suffix(os.path.join(output_dir, 'masksTr'), 'mask1.png')
    train_mask2 = list_files_with_suffix(os.path.join(output_dir, 'masksTr'), 'mask2.png')
    train_mask3 = list_files_with_suffix(os.path.join(output_dir, 'masksTr'), 'mask3.png')

    # Generate CSV for training data
    train_data = {
        'train_images': train_images,
        'train_mask1': train_mask1,
        'train_mask2': train_mask2,
        'train_mask3': train_mask3,
    }
    generate_csv(train_data, os.path.join(output_dir, 'train_data_img.csv'))


def main():
    # Directories for raw dataset and output
    input_dir = 'D:/deep_learning/raw'
    output_dir = 'D:/deep_learning/output'

    preproc_main(input_dir, output_dir)

if __name__ == '__main__':
    main()
