import base64
import os
from io import BytesIO
import PIL.Image

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor


def ReverseImgNorm(x):
    x = x * 0.5 + 0.5
    x *= 255.0
    if isinstance(x, np.ndarray):
        x = x.clip(0, 255).astype(np.uint8)
    else:
        x = torch.clamp(x, 0, 255).to(torch.uint8)
    return x

def ReverseCoordNorm(np_array, size):
    return (np_array + 1) * (size - 1) / 2


def get_rgbs(pred, image_size):
    rgbs = []
    for pred_x, pred_y in pred:
        r = max(0.0, min(pred_x / image_size, 1.0)) 
        b = max(0.0, min(pred_y / image_size, 1.0)) 
        rgbs.append((r, 0, b))
    return rgbs

def get_nonzero_corrs(corrs): 
    mask = torch.all(corrs == 0, dim=1)
    M = torch.sum(mask.flip(0)).item()
    return corrs[:-M] if M != 0 else corrs

def get_viz(view1, view2, pred1, pred2, losses=None, sort=False):
    def gen_plot(view1, view2, pred1, pred2, losses=None):
        view1_img = view1["img"].permute(0, 2, 3, 1).cpu().numpy()
        view2_img = view2["img"].permute(0, 2, 3, 1).cpu().numpy()

        B, image_size, _,  _ = view1_img.shape
        # titles = ["gt", "pred", "conf_plan", "conf_image", "image"]
        titles = ["gt", "pred2", "conf_pred2", "image+correspondences", "image"]
        N = len(titles)
        fig, axes = plt.subplots(B, N, figsize=(30, B*N))
        plt.axis("off")
        
        bs = np.argsort(losses) if losses is not None and sort else range(B)
        idx = 0

        centroids_diff = []

        for b in bs:
            if B == 1:
                view1_img_scaled = ReverseImgNorm(view1_img[b])
                axes[0].imshow(view1_img_scaled)
                gt = get_nonzero_xys(view1["xys"][b].cpu()).numpy()
                axes[0].scatter(gt[:,0], gt[:,1], s=5)
                axes[0].set_title(titles[0])
                
                axes[1].imshow(view1_img_scaled)    
                pred = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                view2_xys = get_nonzero_xys(view2["xys"][b].cpu())
                x_coords = view2_xys[:,0].numpy()
                y_coords = view2_xys[:,1].numpy()
                pred = np.stack((pred[y_coords, x_coords, 0], pred[y_coords, x_coords, 2]), axis=1)
                pred = ReverseCoordNorm(pred, image_size)
                rgbs = get_rgbs(pred, image_size)
                axes[1].scatter(pred[:,0], pred[:,1], s=5, c=rgbs)
                axes[1].set_title(titles[1])   

                conf2 = pred2["conf"][b].detach().cpu().numpy()
                axes[2].imshow(conf2)   
                axes[2].set_title(titles[2]) 

                view2_img_scaled = ReverseImgNorm(view2_img[b])
                axes[3].imshow(view2_img_scaled)   
                image_xys = get_nonzero_xys(view2["xys"][b].cpu()).numpy()   
                axes[3].scatter(image_xys[:,0], image_xys[:,1], s=1, c=rgbs) 
                axes[3].set_title(titles[3])      

                axes[4].imshow(view2_img_scaled)   
                axes[4].set_title(titles[4])   

                if losses is not None:
                    axes[0].set_ylabel(f"{b}. loss: {losses[b]:.6f}")
                else:
                    axes[0].set_ylabel(f"{b}.")

                centroids_diff.append(np.linalg.norm(get_centroid(pred)-get_centroid(gt)))
            else:
                view1_img_scaled = ReverseImgNorm(view1_img[b])
                axes[idx, 0].imshow(view1_img_scaled)
                gt = get_nonzero_xys(view1["xys"][b].cpu()).numpy()
                axes[idx, 0].scatter(gt[:,0], gt[:,1], s=5)
                axes[idx, 0].set_title(titles[0])
                
                axes[idx, 1].imshow(view1_img_scaled)    
                pred = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                view2_xys = get_nonzero_xys(view2["xys"][b].cpu())
                x_coords = view2_xys[:,0].numpy()
                y_coords = view2_xys[:,1].numpy()
                pred = np.stack((pred[y_coords, x_coords, 0], pred[y_coords, x_coords, 2]), axis=1)
                pred = ReverseCoordNorm(pred, image_size)
                rgbs = get_rgbs(pred, image_size)
                axes[idx, 1].scatter(pred[:,0], pred[:,1], s=5, c=rgbs)
                axes[idx, 1].set_title(titles[1])   

                conf2 = pred2["conf"][b].detach().cpu().numpy()
                axes[idx, 2].imshow(conf2)   
                axes[idx, 2].set_title(titles[2]) 

                view2_img_scaled = ReverseImgNorm(view2_img[b])
                axes[idx, 3].imshow(view2_img_scaled)   
                image_xys = get_nonzero_xys(view2["xys"][b].cpu()).numpy()   
                axes[idx, 3].scatter(image_xys[:,0], image_xys[:,1], s=1, c=rgbs) 
                axes[idx, 3].set_title(titles[3])      

                axes[idx, 4].imshow(view2_img_scaled)   
                axes[idx, 4].set_title(titles[4])   
                
                if losses is not None:
                    axes[idx, 0].set_ylabel(f"{b}. loss: {losses[b]:.6f}")
                else:
                    axes[idx, 0].set_ylabel(f"{b}.")

                centroids_diff.append(np.linalg.norm(get_centroid(pred)-get_centroid(gt)))

            idx += 1

        plt.subplots_adjust(hspace=0.0, wspace=0.0)  # Set both to 0 to remove space
        plt.tight_layout()
        return fig, np.array(centroids_diff)
    viz, centroids_diff = gen_plot(view1, view2, pred1, pred2, losses=losses)
    return viz, centroids_diff


