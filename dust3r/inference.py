# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf

import numpy as np
from dust3r.datasets import get_data_loader
from dust3r.utils.image import load_megascenes_augmented_images
from dust3r.utils.viz import get_viz
import PIL
import matplotlib
import os
import io
from io import BytesIO
import base64

def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # ['img', 'xys', 'instance']
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):

            loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None
            # view1=view2: dict("img": Tensor(BCHW=(4,3,244,244)), "true_shape": Tensor(4,2), "instance": list(4), "plan_xy": Tensor(4,2), "image_xy": Tensor(4,2))
            # pred1: dict("pts3d": Tensor(BHWC=(4,224,224,3)), "conf": Tensor(BHW=(4,224,224)))
            # pred2: dict("pts3d_in_other_view": Tensor(BHWC), "conf": Tensor(BHW))

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result

def losses_greater_than_x(losses, threshold):
    if isinstance(losses, list):
        losses = np.array(losses)
    return losses[losses >= threshold].sum()/len(losses)

# @torch.no_grad()
# def inference(pairs, model, device, batch_size=8, verbose=True):
#     if verbose:
#         print(f'>> Inference with model on {len(pairs)} image pairs')
#     result = []

#     # first, check if all images have the same size
#     multiple_shapes = not (check_if_same_size(pairs))
#     if multiple_shapes:  # force bs=1
#         batch_size = 1

#     for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
#         res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
#         result.append(to_cpu(res))

#     result = collate_with_cat(result, lists=multiple_shapes)

#     return result

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Train (heldout)/Test'][test]
    print(f'Building {split} Data loader for dataset')
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        test=test
    )

    print(f"{split} dataset length: {len(loader)}")
    return loader

def make_batches(plan_path, img_path, xys_path, batch_size):
        plan_xys, image_xys = np.load(xys_path)
        pair = load_megascenes_augmented_images((plan_path, img_path), size=512, plan_xys=plan_xys, image_xys=image_xys, augment=False)  
        batches = build_dataset([pair], batch_size, num_workers=4, test=True)
        return batches

@torch.no_grad()
def inference(model, test_criterion, device, epoch, output_dir, log_writer):
    def get_inference_viz(batches, model, criterion, device):
        viz_list = []
        centroids_diff_list = []
        for batch in batches:
            output = loss_of_one_batch(batch, model=model, criterion=criterion, device=device, symmetrize_batch=False, use_amp=False, ret=None)
            loss, loss_details = output["loss"]
            loss_value = sum(loss_details.values())
            viz, centroids_diff = get_viz(output["view1"], output["view2"], output["pred1"], output["pred2"], [loss_value])
            viz_list.append(viz)
            centroids_diff_list.append(centroids_diff)
        return viz_list, centroids_diff_list
    
    pairs_path = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data/intuitive_pairs.txt"
    pairs_info = []
    with open(pairs_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(" /share")
            pairs_info.append((line[0], "/share"+line[1], "/share"+line[2]))

    npx_dir = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data/data_test/coords"
    fig_list = []
    centroids_diff_list = []
    for idx, (npx_num, plan_path, image_path) in enumerate(pairs_info):
        npx_path = os.path.join(npx_dir, f"{int(npx_num):06}.npy")
        batches = make_batches(plan_path, image_path, npx_path, batch_size=1)
        viz, centroids_diff= get_inference_viz(batches, model, test_criterion, device)
        fig_list.append(viz[0])
        log_writer.add_scalar(f"intuitive_centroids_diff_{idx}", centroids_diff[0][0], epoch)
        centroids_diff_list.append(centroids_diff[0][0])
    log_writer.add_scalar("intuitive_centroids_diff", np.mean(centroids_diff_list), epoch)

    pil_images = []
    for fig in fig_list:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pil_images.append(PIL.Image.open(buf))

    # Calculate dimensions
    widths, heights = zip(*(img.size for img in pil_images))
    max_width = max(widths)
    total_height = sum(heights)

    # Create a new image
    stacked_image = PIL.Image.new('RGBA', (max_width, int(total_height * 1.5)))

    # Paste images
    y_offset = 0
    for img in pil_images:
        stacked_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the result
    os.makedirs(os.path.join(output_dir, "intuitive_pairs"), exist_ok=True)
    
    buffer = BytesIO()
    stacked_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    output_html_path = os.path.join(output_dir, "intuitive_pairs", f"intuitive_pairs_epoch_{epoch}.html")
    with open(output_html_path, 'w') as f:
        f.write(f'<img src="data:image/png;base64,{img_str}">')

def demo():
    print("demo")

def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
