import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

import nibabel as nib
import os
import time
import gc
import copy
from pytorch_toolbelt import losses as L

import albumentations as A
from albumentations.pytorch import ToTensorV2


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
force_cudnn_initialization()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def image_transform(img,mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9,1.1), shear=(-5,5), rotate=(-10,10), p=0.3,fit_output=False,keep_ratio=True),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.2),
        A.RandomBrightnessContrast(p=0.4),
    ])
    # Example tensor with shape [1, 128, 128, 32]
    tensor = img.squeeze(0)  # Remove batch dimension, shape now [128, 128, 32]
    mask = mask.squeeze(0)
    #print(tensor.shape,mask.shape)
    img_slices = []
    mask_slices=[]
    for i in range(tensor.shape[2]):  # Iterate over depth slices
        img_slice = tensor[:, :, i].numpy().astype(np.uint8)  
        mask_slice=mask[:,:,i].numpy().astype(np.uint8)
        # Apply the albumentations transform
        transformed_img = transform(image=img_slice)
        transformed_mask= transform(image=mask_slice)
        transformed_img_slice = transformed_img['image']
        transformed_mask_slice=transformed_mask['image']
        # Ensure the transformed slice has the correct shape and type
        img_slices.append(transformed_img_slice)
        mask_slices.append(transformed_mask_slice)
    # Stack the transformed slices along the depth axis
    transformed_img = np.stack(img_slices).T
    transformed_img=torch.tensor(transformed_img).unsqueeze(0).float()
    transformed_mask = np.stack(mask_slices).T
    transformed_mask=torch.tensor(transformed_mask).long()
    #print(transformed_tensor.shape)
    return transformed_img, transformed_mask


class MedicalImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(128, 128, 32), transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        # Load the image and mask
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).unsqueeze(0).float()  # Adding channel dimension (1, H, W, D)
        mask = torch.from_numpy(mask).unsqueeze(0).long()  # Adding channel dimension (1, H, W, D)
        #print(mask.shape)

        # Resize image and mask to target size using interpolation
        image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.target_size, mode='nearest').squeeze(0).long()

        # Remove the channel dimension from the mask
        mask = mask.squeeze(0)  # Now the mask is of shape (H, W, D)
        image,mask = image_transform(image,mask)

        mask = torch.clamp(mask, 0, 1)

        #print(image.shape)
        return image, mask



class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool3d(2)
        
        self.bottleneck = conv_block(256, 512)
        
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.conv_last = nn.Conv3d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_last(dec1)

images_dir = '/home/bdbotond/python/deep_learning/test/ASCENT/data/ACDC/raw/imagesTr'
masks_dir = '/home/bdbotond/python/deep_learning/test/ASCENT/data/ACDC/raw/labelsTr'

full_dataset = MedicalImageDataset(images_dir, masks_dir)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])


train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader= DataLoader(val_dataset, batch_size=4, shuffle=True)

# Initialize the model, loss function, and optimizer
model = UNet(in_channels=1, out_channels=2)  
loss_function =L.SoftCrossEntropyLoss()
#loss_function = nn.CrossEntropyLoss()  
model.to(device)

num_epochs =20


def train_model(model,loss_function,lr):
    model=model
    loss_function=loss_function
    optimizer=optim.Adam(model.parameters(), lr=lr)
    since=time.time()

    best_error=100
    #best_model=copy.deepcopy(model.state_dict())
    train_loss=[]
    val_loss=[]
    for epoch in range(num_epochs):
        model.train()
        running_loss=0
        iou_train=0
        for images, masks in train_dataloader:
            
            images=images.to(device)
            masks=masks.to(device)
            outputs = model(images)
            optimizer.zero_grad()
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iou_train+=(calc_iou(outputs,masks))

            #train_loss.append(loss.item())
        model.eval()
        validation_loss=0
        iou_val=0
        with torch.no_grad():
            for images,labels in val_dataloader:
                images=images.to(device)
                labels=labels.to(device)
                output_val=model(images)
                val_loss=loss_function(output_val,labels)
                validation_loss += val_loss.item()
                iou_val+=(calc_iou(outputs,masks))

                #val_loss.append(val_loss)
        t_loss=running_loss/len(val_dataloader)
        v_loss=validation_loss/len(val_dataloader)
        it=iou_train/len(val_dataloader)
        iv=iou_val/len(val_dataloader)
        if v_loss<=best_error:
                best_error=v_loss
                best_epoch=epoch
                print(best_error,best_epoch)
                best_model=copy.deepcopy(model.state_dict())
                torch.save(best_model,'./u_net_tacc_lr'+str(lr)+'.pt')
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {t_loss},Train IoU: {it},Validation Loss: {v_loss},Validation IoU:{iv}")

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    time_elapsed=time.time()-since
    print(f"Training time was:[{time_elapsed}]")

train_model(model,loss_function,1e-3)