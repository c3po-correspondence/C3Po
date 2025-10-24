import numpy as np
import torch
from dust3r.datasets.utils.transforms import *

from .c3 import C3  # noqa


def collate_fn(batch):  
    max_corrs_len = max(item[0]["corrs"].shape[0] for item in batch)

    view1_imgs, view2_imgs = [], []
    view1_corrs, view2_corrs = [], []
    view1_instances, view2_instances = [], []
    
    for view1, view2 in batch: 
        # view1_imgs.append(torch.squeeze(torch.Tensor(view1["img"]), 0))
        # view2_imgs.append(torch.squeeze(torch.Tensor(view2["img"]), 0))
        view1_imgs.append(torch.as_tensor(view1["img"]).squeeze(0))
        view2_imgs.append(torch.as_tensor(view2["img"]).squeeze(0))
        view1_instances.append(view1["instance"])
        view2_instances.append(view2["instance"])

        pad_len1 = max_corrs_len - view1["corrs"].shape[0]
        pad_len2 = max_corrs_len - view2["corrs"].shape[0]

        view1_corrs.append(torch.from_numpy(np.pad(view1["corrs"], ((0, pad_len1), (0, 0)), mode="constant", constant_values=0)))
        view2_corrs.append(torch.from_numpy(np.pad(view2["corrs"], ((0, pad_len2), (0, 0)), mode="constant", constant_values=0)))

    view1_imgs = torch.stack(view1_imgs)
    view2_imgs = torch.stack(view2_imgs)
    view1_corrs = torch.stack(view1_corrs)
    view2_corrs = torch.stack(view2_corrs)

    view1_batch = dict(
        img=view1_imgs, 
        corrs=view1_corrs,
        instance=view1_instances,
    )
    view2_batch = dict(
        img=view2_imgs,
        corrs=view2_corrs,
        instance=view2_instances,
    )

    return view1_batch, view2_batch

def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True, test=None):
    import torch
    from croco.utils.misc import get_rank, get_world_size

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader