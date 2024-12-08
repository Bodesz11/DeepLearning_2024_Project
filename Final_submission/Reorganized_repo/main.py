import json
import torch
from DataLoader import prepare_dataloaders
from Model import get_model
from Train_model import train_model
from Evaluate_model import evaluate_model
import torch.optim as optim
from pytorch_toolbelt import losses as L


def main():
    # Load config
    with open('config.json') as f:
        config = json.load(f)

    # Prepare DataLoaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        f"{config['input_dir']}/train_data.csv",
        f"{config['input_dir']}/test_data.csv",
        batch_size=config["batch_size"]
    )

    # Initialize model
    model = get_model(config["model_type"])
    loss_function = L.DiceLoss(mode='multilabel')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode="exp_range")

    # Train or Load Model
    if config["train"]:
        train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, num_epochs=config["num_epochs"])
        torch.save(model.state_dict(), config["model_path"])
    else:
        model.load_state_dict(torch.load(config["model_path"]))

    # Evaluate Model
    evaluate_model(model, test_loader, config["output_dir"])


if __name__ == "__main__":
    main()
