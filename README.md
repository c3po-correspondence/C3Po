
# C3Po: Cross-View Cross-Modality Correspondence by Pointmap Prediction
### NeurIPS 2025 [Datasets and Benchmarks Track]

[Project Website](https://c3po-correspondence.github.io/) | [Paper (coming soon)](https://c3po-correspondence.github.io/) | [Dataset](https://huggingface.co/datasets/kwhuang/C3)

## Contents
* [Install](#install)
* [Dataset](#dataset)
* [Checkpoint](#checkpoint)
* [Demo](#demo)
* [Training](#training)
* [Evaluation](#evaluation)
* [Citation](#citation)


## Install

1. Clone C3Po.
```bash
git clone --recursive git@github.com:c3po-correspondence/C3Po.git
cd C3Po
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n c3po python=3.11 cmake=3.14.0
conda activate c3po 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121  # use the correct version for you
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add pyrender, used to render depthmap in some datasets preprocessing
# - add required packages for visloc.py
pip install -r requirements_optional.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

## Dataset

The full dataset is available on [`HuggingFace`](https://huggingface.co/datasets/kwhuang/C3). It provides a complete overview of the dataset, including its structure, the contents of each file, and code for visualizing photo-plan pairs and their correspondences and camera poses.

## Checkpoint

Pre-trained model weights:
[`ckpt.pth`](https://drive.google.com/drive/folders/1OoJrtdfjYZhzvlmF9eKpptyWPbXvjofh?usp=sharing)

To run the demo, download the model checkpoint and save the weights in `demo/`.

## Demo

After saving the model weights in `demo/`, run `demo.ipynb` to visualize plan-photo pairs, predicted correspondences, and ground truth correspondences. 


## Training
Download [DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth](https://github.com/naver/dust3r?tab=readme-ov-file#checkpoints).
```bash
torchrun --nproc_per_node 8 train.py \
--train_dataset="C3(data_dir='/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/c3po/', image_dir='/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive/', split='train', resolution=[(512, 512)], augmentation_factor=3)" \
--val_dataset="C3(data_dir='/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/c3po/', image_dir='/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive/', split='val', resolution=[(512, 512)], augmentation_factor=1)" \
--test_dataset="C3(data_dir ='/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/c3po/', image_dir='/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive/', split='test', resolution=[(512, 512)], augmentation_factor=1)" \
--train_criterion="ConfLoss(CorrespondenceLoss(L21), alpha=0.2)" \
--test_criterion="CorrespondenceLoss(L21)" \
--model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
--pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
--lr=1e-04 --min_lr=1e-06 --warmup_epochs=3 --epochs=500 --train_batch_size=2 --test_batch_size=1 --accum_iter=3 \
--save_freq=1 --keep_freq=1 --eval_freq=1 --num_workers=12 \
--output_dir="checkpoints/c3po"
```


## Evaluation
```bash
python c3po_inference.py
```


## Citation
```
@inproceedings{
      huang2025c3po,
      title={C3Po: Cross-View Cross-Modality Correspondence by Pointmap Prediction}, 
      author={Huang, Kuan Wei and Li, Brandon and Hariharan, Bharath and Snavely, Noah},
      booktitle={Advances in Neural Information Processing Systems},
      volume={38},
      year={2025}
}
```

