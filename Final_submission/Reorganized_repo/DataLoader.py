import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
import numpy as np

# -------------------------------
# Device Configuration
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Image Transformation Functions
# -------------------------------

def img_trf_train(img_path, mask_path1, mask_path2, mask_path3):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    msk1 = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
    msk2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
    msk3 = cv2.imread(mask_path3, cv2.IMREAD_GRAYSCALE)

    max_side = max(img.shape[0], img.shape[1], msk1.shape[0], msk1.shape[1])

    aug = A.Compose([
        A.PadIfNeeded(min_height=max_side, min_width=max_side, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
        A.Affine(scale=(0.8, 1.1), shear=(-20, 20), rotate=(-20, 20), p=0.8, fit_output=True, keep_ratio=True, mode=0, cval=0),
        A.Resize(128, 128),
    ], is_check_shapes=False)

    masks = np.stack([msk1, msk2, msk3], axis=0).T
    augmented = aug(image=img, mask=masks)

    image_padded = torch.tensor(augmented['image'].T)
    mask_padded = torch.tensor(augmented['mask'].T)
    return image_padded, mask_padded


def img_trf_val(img_path, mask_path1, mask_path2, mask_path3):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    msk1 = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
    msk2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
    msk3 = cv2.imread(mask_path3, cv2.IMREAD_GRAYSCALE)

    max_side = max(img.shape[0], img.shape[1], msk1.shape[0], msk1.shape[1])

    aug = A.Compose([
        A.PadIfNeeded(min_height=max_side, min_width=max_side, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
        A.Resize(128, 128),
    ], is_check_shapes=False)

    masks = np.stack([msk1, msk2, msk3], axis=0).T
    augmented = aug(image=img, mask=masks)

    image_padded = torch.tensor(augmented['image'].T)
    mask_padded = torch.tensor(augmented['mask'].T)
    return image_padded, mask_padded

# -------------------------------
# Dataset Classes
# -------------------------------


class CustomDataset(Dataset):
    def __init__(self, frame, transform_function):
        self.frame = frame
        self.image_files = frame['train_images'].tolist()
        self.mask_files1 = frame['train_mask1'].tolist()
        self.mask_files2 = frame['train_mask2'].tolist()
        self.mask_files3 = frame['train_mask3'].tolist()
        self.transform_function = transform_function

    def __getitem__(self, index):
        img, mask = self.transform_function(
            str(self.image_files[index]),
            str(self.mask_files1[index]),
            str(self.mask_files2[index]),
            str(self.mask_files3[index])
        )
        return img, mask

    def __len__(self):
        return len(self.frame)

# -------------------------------
# DataLoader Preparation
# -------------------------------

def prepare_dataloaders(train_csv, test_csv, batch_size=16):
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_df, img_trf_train)
    val_dataset = CustomDataset(val_df, img_trf_val)
    test_dataset = CustomDataset(test_data, img_trf_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
