
# C3Po: Cross-View Cross-Modality Correspondence by Pointmap Prediction

⚠️ **Work in Progress** ⚠️  
This repository is currently under development.  
Only the demo code is available at this time — the full source code will be released soon. Stay tuned!

## Get Started

### Installation

1. Clone C3Po.
```bash
git clone --recursive git@github.com:kwhuang88228/C3Po.git
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

### Checkpoints

Pre-trained model weight:
[`ckpt.pth`](https://drive.google.com/drive/folders/1OoJrtdfjYZhzvlmF9eKpptyWPbXvjofh?usp=sharing)

Download weight to demo/

### Demo

Run demo.ipynb


### Full Dataset

Full dataset on HuggingFace [`C3`](https://huggingface.co/datasets/kwhuang/C3)

