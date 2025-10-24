import math
from copy import copy, deepcopy

import torch
import torch.nn as nn
from dust3r.inference import find_opt_scaling, get_pred_pts3d
from dust3r.utils.geometry import (geotrf, get_joint_pointcloud_center_scale,
                                   get_joint_pointcloud_depth, inv,
                                   normalize_pointcloud)


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}, {b.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        dist = torch.norm(a - b, dim=-1)  # normalized L2 distance
        return dist


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details
    

def CoordNorm(array, image_dim):
    return torch.tensor(array, dtype=torch.float) / (image_dim - 1)

def get_last_nonzero_indices(corrs):
        last_nonzero_indices = []
        zero_tensor = torch.zeros(2, device=corrs.device if isinstance(corrs, torch.Tensor) else 'cuda')
        for batch in corrs:
            # Iterate backwards
            for idx in range(len(batch) - 1, -1, -1):
                if not torch.all(batch[idx] == zero_tensor):
                    last_nonzero_indices.append(idx + 1)
                    break
        return last_nonzero_indices

class CorrespondenceLoss(Criterion, MultiLoss):
    def __init__(self, criterion):
        super().__init__(criterion)

    def get_name(self):
        return f'CorrespondenceLoss()'

    def get_gt_pts2_and_mask2(self, plan_corrs_padded, photo_corrs_padded, last_nonzero_indices, size):
        B, H, W, _ = size
        device = plan_corrs_padded.device if isinstance(plan_corrs_padded, torch.Tensor) else 'cuda'
        gt_pts2 = torch.zeros((B, H, W, 2), dtype=torch.float).to(device)
        mask2 = torch.zeros((B, H, W), dtype=torch.bool).to(device)

        for b, (plan_corrs, photo_corrs, last_idx) in enumerate(zip(plan_corrs_padded, photo_corrs_padded, last_nonzero_indices)):
            plan_corrs = plan_corrs[:last_idx].long()
            photo_corrs = photo_corrs[:last_idx].long()
            
            gt_pts2[b, photo_corrs[:, 1], photo_corrs[:, 0]] = CoordNorm(plan_corrs, H)
            mask2[b, photo_corrs[:, 1], photo_corrs[:, 0]] = True

        return gt_pts2, mask2
    
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, **kw):
        pred2_pts3d = pred2["pts3d_in_other_view"]  
        pred2_pts3d_x = pred2_pts3d[..., 0:1]  
        pred2_pts3d_z = pred2_pts3d[..., 2:3]  
        pred_pts2 = torch.cat([pred2_pts3d_x, pred2_pts3d_z], dim=-1) 

        plan_corrs_padded = gt1["corrs"]            
        photo_corrs_padded = gt2["corrs"]             
        last_nonzero_indices = get_last_nonzero_indices(photo_corrs_padded)   
        gt_pts2, mask2 = self.get_gt_pts2_and_mask2(plan_corrs_padded, photo_corrs_padded, last_nonzero_indices, pred_pts2.size())

        return gt_pts2, pred_pts2, mask2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts2, pred_pts2, mask2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
            
        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])  
        self_name = type(self).__name__
        details = {self_name + '_pts3d_2': float(l2.mean())}
        return ((l2, mask2)), (details | monitoring)
        
class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        (loss2, msk2), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)

        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2]) 
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        return conf_loss2, dict(conf_loss2=float(conf_loss2), **details)