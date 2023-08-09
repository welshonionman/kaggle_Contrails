import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CustomModel(nn.Module):
    def __init__(self, model_arch, backbone, in_chans, target_size, weight):
        super().__init__()

        self.model = smp.create_model(
            model_arch,
            encoder_name=backbone,
            encoder_weights=weight,
            in_channels=in_chans,
            classes=target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.model(image)
        return output


def build_model(model_arch, backbone, in_chans, target_size, weight="imagenet", dataparallel=True):
    print('model_arch: ', model_arch)
    print('backbone: ', backbone)
    model = CustomModel(model_arch, backbone, in_chans, target_size, weight)

    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataparallel:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    return model


def load_model(pth_path):
    pth = torch.load(f'{pth_path}')

    model = build_model(pth["model_arch"], pth["backbone"], pth["in_chans"], pth["target_size"], weight=None, dataparallel=False)
    model.load_state_dict(pth['model'])
    thresh = pth['thresh']
    dice_score = pth['dice_score']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, dice_score, thresh
