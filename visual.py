from __future__ import annotations
import os.path
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import zoom
from medpy import metric
from loguru import logger
from typing import Any
from PIL import Image

SYNAPSE_CLASS_COLOR_MAP = [
    [0, 0, 0],
    [30, 144, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [128, 0, 255],
    [255, 128, 0]
]
def visual_single_volume(
    model: nn.Module,
    volume: torch.Tensor,
    label: torch.Tensor,
    case_name: str,
    num_classes: int = 9,
    patch_size: list[int] = (224, 224),
    deep_supervision: bool = False,
    device: str = "cuda:0",
    output_folder: str = "visual",
    **kwargs: Any
) -> list[tuple[float, float, float, float]]:
    """
    input tensor shape:
        image: [1, depth, height, width]; label: [1, depth, height, width]

    output list: [(dice, hd95, jaccard, asd), ...]
    """
    volume = volume.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    logger.info("Predicting...")
    prediction = np.zeros_like(label)
    for depth in tqdm(range(volume.shape[0])):
        image_slice = volume[depth, :, :]
        label_slice = label[depth, :, :]
        h, w = image_slice.shape
        if h != patch_size[0] or w != patch_size[1]:
            image_slice = zoom(image_slice, (patch_size[0] / h, patch_size[1] / w), order=3)
            label_slice = zoom(label_slice, (patch_size[0] / h, patch_size[1] / w), order=3)

        if kwargs.get("norm_x_transform", None) is not None:
            input = kwargs.get("norm_x_transform")(image_slice)
        else:
            input = torch.from_numpy(image_slice).unsqueeze(0)
        input = input.unsqueeze(0).float().to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input)
            if deep_supervision: outputs = sum(outputs)

            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
        
            image_numpy = np.abs(image_slice) * 255
            image_ori = np.array(image_numpy, dtype=np.uint8)
            
            image_ori = Image.fromarray(image_ori)
            image_ori = image_ori.convert("RGB")
            image_ori = np.array(image_ori)
            visual_pred = image_ori.copy()
            visual_gd = image_ori.copy()
            for i in range(1, 9):
                visual_pred[out == i, :] = SYNAPSE_CLASS_COLOR_MAP[i]
                visual_gd[label_slice == i, :] = SYNAPSE_CLASS_COLOR_MAP[i]

            image_ori = Image.fromarray(image_ori)
            visual_gd = Image.fromarray(visual_gd)
            visual_pred = Image.fromarray(visual_pred)
            paste_image = Image.new("RGB", ((224 * 3), 224))

            paste_image.paste(image_ori, (0, 0))
            paste_image.paste(visual_gd, (214*1, 0))
            paste_image.paste(visual_pred, (214*2, 0))
            paste_image.save(f"visual/image{depth}.png")
        
            





def inference(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int = 9,
    patch_size: list[int] = (224, 224),
    deep_supervision: bool = False,
    output_folder: str = "visual",
    device: str = "cuda:0",
    **kwargs: Any,
) -> None:
    logger.info(f"Testing iterations: {len(dataloader)}")
    os.makedirs(output_folder, exist_ok=True)
    
    # 控制处理那一卷
    # 4 卷效果比较好 106slice左右对比最好
    flag = 4
    i = 0
    for sample in tqdm(dataloader):
        if i == flag:
            image, label, case_name = sample["image"], sample["label"], sample['case_name'][0]
            visual_single_volume(
                model=model,
                volume=image,
                label=label,
                num_classes=num_classes,
                case_name=case_name,
                patch_size=patch_size,
                deep_supervision=deep_supervision,
                device=device,
                output_folder=output_folder,
                **kwargs
            )
        i = i + 1

def get_model(ckpt: str, **kwargs: Any) -> nn.Module:
    from model import build_model

    state_dict = OrderedDict()
    for k, v in torch.load(ckpt, map_location="cpu")["state_dict"].items():
        state_dict[k.replace("_model.", "")] = v

    model = build_model(**kwargs)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model checkpoint: {ckpt}")
    return model

def test_acdc(ckpt: str) -> None:
    from dataset_acdc import ACDCDataset
    from torchvision.transforms import transforms

    norm_x_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    output_folder = "testing_acdc"
    logger.add(os.path.join(output_folder, "testing.log"))
    data_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    device = "cuda:0"
    model = get_model(ckpt=ckpt, in_channels=3, num_classes=4).to(device)
    dataset = ACDCDataset(base_dir=os.path.join(data_root, "dataset/acdc"), split="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inference(
        model=model,
        dataloader=dataloader,
        num_classes=4,
        output_folder=output_folder,
        device=device,
        norm_x_transform=norm_x_transform,
    )

def test_synapse(ckpt: str) -> None:
    from dataset_synapse import SynapseDataset
    from torchvision.transforms import transforms

    norm_x_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    output_folder = "testing_synapse"
    logger.add(os.path.join(output_folder, "testing.log"))

    device = "cuda:0"
    model = get_model(ckpt=ckpt, in_channels=3, num_classes=9).to(device)
    data_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset = SynapseDataset(base_dir=os.path.join(data_root, "dataset/synapse/test_vol"), split="test_vol")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inference(
        model=model,
        dataloader=dataloader,
        num_classes=9,
        output_folder=output_folder,
        device=device,
        norm_x_transform=norm_x_transform,
    )