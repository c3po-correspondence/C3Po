import csv
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as tvf
from dust3r.datasets import get_data_loader, C3
from dust3r.model import AsymmetricCroCo3DStereo, inf
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 300000000
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

resolution = (512, 512)

def CoordNorm(array, image_dim):
    return torch.tensor(2 * (array / (image_dim - 1)) - 1, dtype=torch.float)

def CoordDenorm(tensor):
    return (tensor + 1) / 2.0

def correspondence_rmse(pred, gt, per_image=False):
    if per_image:
        return torch.sqrt(torch.mean((pred - gt) ** 2, dim=1))
    return torch.sqrt(torch.mean((pred - gt) ** 2))

def process(view1, view2, pred1, pred2):
    plan_corrs = view1["corrs"][0]            
    photo_corrs = view2["corrs"][0]    
        
    gt = CoordNorm(plan_corrs, resolution[0])

    pred_xyz = pred2["pts3d_in_other_view"]  
    pred_x = pred_xyz[..., 0:1] 
    pred_z = pred_xyz[..., 2:3] 
    pred = torch.cat([pred_x, pred_z], dim=-1)  
    pred = pred[0, photo_corrs[:, 1], photo_corrs[:, 0]]

    assert pred.shape == gt.shape   

    return pred, gt


# === Load Dataset ===
print("Loading data...")
dataset = C3(
    data_dir ='C3/geometric/', 
    image_dir='C3/visual/', split='test', 
    resolution=[resolution], 
    augmentation_factor=1
)

dataloader = get_data_loader(
    dataset,
    batch_size=1,
    num_workers=4,
    shuffle=False,
    test=True
)

# === Load Model and Weights ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)

weights_path = "demo/ckpt.pth"
ckpt = torch.load(weights_path, map_location=device)
model.load_state_dict(ckpt['model'], strict=False)

model.to(device)
model.eval()


preds = []
gts = []

# === Inference Loop ===
for i, batch in tqdm(enumerate(dataloader)):
    view1, view2 = batch

    # Remove unused keys
    view1.pop("instance", None)
    view2.pop("instance", None)

    # Move tensors to device
    for view in (view1, view2):
        for key, val in view.items():
            view[key] = val.to(device, non_blocking=True)

    with torch.no_grad():
        pred1, pred2 = model(view1, view2)

        # Process predictions
        pred, gt = process(view1, view2, pred1, pred2)

        preds.append(pred.cpu().numpy())
        gts.append(gt.cpu().numpy())

    if i == 200:
        break

rmse = correspondence_rmse(
    CoordDenorm(torch.tensor(np.concatenate(preds))), 
    CoordDenorm(torch.tensor(np.concatenate(gts)))
)
print(f"Correspondence RMSE: {rmse.item():.4f}")





