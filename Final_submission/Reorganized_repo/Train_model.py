import torch
import wandb

def train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, num_epochs=10, device='cuda'):
    model.to(device)
    wandb.init(project="segmentation_training")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss})
