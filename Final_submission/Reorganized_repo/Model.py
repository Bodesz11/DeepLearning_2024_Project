import segmentation_models_pytorch as smp


def get_model(model_type, in_channels=1, classes=3):
    models = {
        "Unet": smp.Unet,
        "MAnet": smp.MAnet,
        "DeepLabV3": smp.DeepLabV3,
        "Linknet": smp.Linknet,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    model = models[model_type](
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
        activation='softmax'
    )

    return model
