import torch
import cv2
import os

def evaluate_model(model, test_loader, output_dir, device='cuda'):
    model.to(device)
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float)
            outputs = model(images).cpu().numpy()

            output_path = os.path.join(output_dir, f"output_{i}.png")
            cv2.imwrite(output_path, outputs[0, 0].T * 255)
            print(f"Saved: {output_path}")
