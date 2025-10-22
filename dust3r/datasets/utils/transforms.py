# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
import math
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

# define the standard image transforms
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])