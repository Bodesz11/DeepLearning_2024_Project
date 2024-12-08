# Imports
import torch
import matplotlib.pyplot as plt


def plot_data(tensor: torch.tensor, batch_index: torch.tensor):
    """
    Visualizes slices from a predicted 3D tensor (e.g., model outputs).

    Args:
        - tensor (torch.Tensor): The predicted output tensor with shape (B, C, H, W, D),
                                 where B is the batch size, C is the number of channels,
                                 and (H, W, D) are the spatial dimensions.
        - batch_index (int): The index of the batch element to visualize.

    Returns:
        - None: Displays the plotted slices.
    """
    channel_index = 1  # Select the second channel (usually representing the prediction)

    # Extract the specific batch and channel
    data = tensor[batch_index, channel_index]  # Shape: [H, W, D]

    # Plot a few slices from this data
    num_slices_to_plot = 5  # Number of slices to visualize
    slice_indices = torch.linspace(0, data.shape[2] - 1, num_slices_to_plot).long()

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(slice_indices):
        plt.subplot(1, num_slices_to_plot, i + 1)
        plt.imshow(data[:, :, idx], cmap='gray')
        plt.title(f"Slice {idx.item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_mask(tensor, batch_index):
    """
    Visualizes slices from a 3D ground truth tensor (e.g., segmentation masks).

    Args:
        - tensor (torch.Tensor): The ground truth tensor with shape (B, H, W, D),
                                 where B is the batch size and (H, W, D) are the spatial dimensions.
        - batch_index (int): The index of the batch element to visualize.

    Returns:
        - None: Displays the plotted slices.
    """
    # Extract the specific batch
    data = tensor[batch_index]  # Shape: [H, W, D]

    # Plot a few slices from this data
    num_slices_to_plot = 5  # Number of slices to visualize
    slice_indices = torch.linspace(0, data.shape[2] - 1, num_slices_to_plot).long()

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(slice_indices):
        plt.subplot(1, num_slices_to_plot, i + 1)
        plt.imshow(data[:, :, idx], cmap='gray')
        plt.title(f"Slice {idx.item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
