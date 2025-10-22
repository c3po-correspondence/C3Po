# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from dust3r.datasets.utils.transforms import *
from PIL import ExifTags
from PIL.ImageOps import exif_transpose
import matplotlib.pyplot as plt

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import warnings

import cv2  # noqa

PIL.Image.MAX_IMAGE_PIXELS = 300000000
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.simplefilter("ignore", PIL.Image.DecompressionBombWarning)


try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False


def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def resize_and_pad(img, coords, size, is_image):
    W, H = img.size

    ratio = min(size / W, size / H)
    W_target = int(W * ratio)
    H_target = int(H * ratio)

    if ratio > 1:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    img_resized = img.resize((W_target, H_target), interp)

    img_resized_padded = PIL.Image.new("RGB", (size, size), (0, 0, 0))
    img_resized_padded.paste(img_resized, ((size - W_target) // 2, (size - H_target) // 2))

    resized_coords = coords * ratio
    offset = np.array([0, (size - H_target) // 2]) if W_target > H_target else np.array([(size - W_target) // 2, 0])
    updated_coords = resized_coords + offset
    if is_image:
        updated_coords = np.clip(updated_coords, 0, size-1)
        
    return img_resized_padded, updated_coords

def get_scaled_plan(im):
    max_h = 500
    # im.size = (w, h)
    plan_scale = max_h / im.size[1]
    scaled_plan = im.resize((round(im.size[0] * plan_scale), max_h))
    return scaled_plan

def crop_outlier_xys(plan_xys, image_xys, size, pair, plan, img):
    mask = np.all((plan_xys >= 0) & (plan_xys <= size - 1), axis=1).astype(int)
    plan_xys_cropped = plan_xys[mask == 1]
    image_xys_cropped = image_xys[mask == 1]

    return plan_xys_cropped, image_xys_cropped

def get_exif_orientation(image):
    exif = image._getexif()
    if exif:
        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag) == 'Orientation':
                return value
    return 1  # default (no rotation)

def transform_points(points, orientation, width, height):
    transformed = []
    for x, y in points:  
        if orientation == 2:  # Horizontal flip
            new_x, new_y = width - x, y
        elif orientation == 3:  # Rotate 180
            new_x, new_y = width - x, height - y
        elif orientation == 4:  # Vertical flip
            new_x, new_y = x, height - y
        elif orientation == 5:  # Vertical flip + rotate 90 CW
            new_x, new_y = y, x
        elif orientation == 6:  # Rotate 270 CW
            new_x, new_y = height - y, x
        elif orientation == 7:  # Horizontal flip + rotate 90 CW
            new_x, new_y = height - y, width - x
        elif orientation == 8:  # Rotate 90 CW
            new_x, new_y = y, width - x
        else:
            new_x, new_y = x, y
        transformed.append((new_x, new_y))
    return np.array(transformed)

def random_crop_custom(image, keypoints, crop_range=(0.95, 1.0)):
    orig_width, orig_height = image.size
    rand_scale = random.uniform(*crop_range)

    crop_w = int(orig_width * rand_scale)
    crop_h = int(orig_height * rand_scale)

    max_left = orig_width - crop_w
    max_top = orig_height - crop_h

    left = random.randint(0, max_left)
    top = random.randint(0, max_top)
    right = left + crop_w
    bottom = top + crop_h

    cropped_image = image.crop((left, top, right, bottom))
    cropped_keypoints = keypoints - np.array([[left, top]])

    return cropped_image, cropped_keypoints

def rotate_image_and_keypoints(image, keypoints, angle_option=None):
    if angle_option is None:
        angle_option = np.random.randint(0, 4)
    angle_degrees = angle_option * 90
    
    orig_w, orig_h = image.size
    rotated_image = image.rotate(angle_degrees, expand=True)
    
    # Create a copy of the keypoints to avoid modifying the original
    rotated_keypoints = keypoints.copy()
    
    if angle_option == 0: 
        pass
    elif angle_option == 1:
        x, y = rotated_keypoints[:, 0].copy(), rotated_keypoints[:, 1].copy()
        rotated_keypoints[:, 0] = y
        rotated_keypoints[:, 1] = orig_w - x
    elif angle_option == 2: 
        rotated_keypoints[:, 0] = orig_w - rotated_keypoints[:, 0]
        rotated_keypoints[:, 1] = orig_h - rotated_keypoints[:, 1]
    elif angle_option == 3:
        x, y = rotated_keypoints[:, 0].copy(), rotated_keypoints[:, 1].copy()
        rotated_keypoints[:, 0] = orig_h - y
        rotated_keypoints[:, 1] = x
    
    return rotated_image, rotated_keypoints

def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        # if "plan" in path:
        #     img = get_scaled_plan(img)
        #     img = resize_and_pad(img, "", size)
        # else:
        #     img = resize_and_pad(img, "", size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def load_megascenes_augmented_images(pair, size, plan_xys, image_xys, augment, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    plan_path, img_path = pair
    image_views = []

    plan = PIL.Image.open(plan_path)
    if plan_path.lower().endswith(".gif"):
        plan.seek(plan.n_frames // 2)
    else:
        plan_orientation = get_exif_orientation(plan)
        plan_xys = transform_points(plan_xys, plan_orientation, plan.size[0], plan.size[1])
    plan = exif_transpose(plan).convert('RGB')

    img = PIL.Image.open(img_path)
    if img_path.lower().endswith(".gif"):
        img.seek(img.n_frames // 2)
    else:
        img_orientation = get_exif_orientation(img)
        image_xys = transform_points(image_xys, img_orientation, img.size[0], img.size[1])
    img = exif_transpose(img).convert('RGB')

    plan_W1, plan_H1 = plan.size
    img_W1, img_H1 = img.size
    scaled_plan = get_scaled_plan(plan)

    if augment:
        scaled_plan, plan_xys = random_crop_custom(scaled_plan, plan_xys, crop_range=(0.95, 1.0))
        scaled_plan, plan_xys = rotate_image_and_keypoints(scaled_plan, plan_xys)
        transform = ColorJitter
    else:
        transform = ImgNorm
    plan_updated, plan_xys_updated = resize_and_pad(scaled_plan, plan_xys, size, is_image=False)
    img_updated, image_xys_updated = resize_and_pad(img, image_xys, size, is_image=True)

    plan_xys_updated, image_xys_updated = crop_outlier_xys(plan_xys_updated, image_xys_updated, size, pair, plan_updated, img_updated)

    plan_W2, plan_H2 = plan_updated.size
    img_W2, img_H2 = img_updated.size

    if verbose:
        print(f' - adding {plan_path} with resolution {plan_W1}x{plan_H1} --> {plan_W2}x{plan_H2}')
        print(f' - adding {img_path} with resolution {img_W1}x{img_H1} --> {img_W2}x{img_H2}')
    image_views.append(
        dict(
            img=transform(plan_updated)[None], 
            true_shape=np.int32([plan_updated.size[::-1]]), 
            idx=len(image_views), 
            instance=str(len(image_views)), 
            xys=np.int32(plan_xys_updated),
            # path=plan_path
        )
    )
    image_views.append(
        dict(
            img=ImgNorm(img_updated)[None], 
            true_shape=np.int32([img_updated.size[::-1]]), 
            idx=len(image_views), 
            instance=str(len(image_views)), 
            xys=np.int32(image_xys_updated),
            # path=img_path
        )
    )

    if verbose:
        print(f' (Found {len(image_views)} images)')
    return image_views


if __name__ == "__main__":
    pass


