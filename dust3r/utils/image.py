import os

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as tvf
from dust3r.datasets.utils.transforms import *
from PIL import ExifTags
from PIL.ImageOps import exif_transpose
import matplotlib.pyplot as plt

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import warnings

import cv2  # noqa

Image.MAX_IMAGE_PIXELS = 300000000
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False


def resize_and_pad(img, corrs, size, is_photo):
    W, H = img.size
    ratio = min(size / W, size / H)
    W_target = int(W * ratio)
    H_target = int(H * ratio)

    # Choose interpolation based on scaling direction
    interp = Image.LANCZOS if ratio > 1 else Image.BICUBIC

    # Resize and create padded image
    img_resized = img.resize((W_target, H_target), interp)
    img_updated = Image.new("RGB", (size, size), (0, 0, 0))
    w_offset = (size - W_target) // 2
    h_offset = (size - H_target) // 2
    img_updated.paste(img_resized, (w_offset, h_offset))

    # Apply same transformations to correspondences
    offset = np.array([0, h_offset]) if W_target > H_target else np.array([w_offset, 0])
    corrs_updated = corrs * ratio + offset

    if is_photo:
        corrs_updated = np.clip(corrs_updated, 0, size-1)

    return img_updated, corrs_updated

def crop_outlier_corrs(plan_corrs, photo_corrs, size, pair, plan, img):
    mask = np.all((plan_corrs >= 0) & (plan_corrs <= size - 1), axis=1).astype(int)
    plan_corrs_cropped = plan_corrs[mask == 1]
    photo_corrs_cropped = photo_corrs[mask == 1]

    return plan_corrs_cropped, photo_corrs_cropped

def get_exif_orientation(photo):
    exif = photo._getexif()
    if exif:
        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag) == 'Orientation':
                return value
    return 1  # default (no rotation)

def transform_correspondences(corrs, orientation, width, height):
    transformed = []
    for x, y in corrs:  
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

def random_crop(image, keypoints, crop_range=(0.95, 1.0)):
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

def random_rotate(plan, corrs, angle_option=None):
    if angle_option is None:
        angle_option = np.random.randint(0, 4)
    angle_degrees = angle_option * 90

    orig_w, orig_h = plan.size
    plan_rotated = plan.rotate(angle_degrees, expand=True)

    corrs_rotated = corrs.copy()
    
    if angle_option == 0: 
        pass
    elif angle_option == 1:
        x, y = corrs_rotated[:, 0].copy(), corrs_rotated[:, 1].copy()
        corrs_rotated[:, 0] = y
        corrs_rotated[:, 1] = orig_w - x
    elif angle_option == 2: 
        corrs_rotated[:, 0] = orig_w - corrs_rotated[:, 0]
        corrs_rotated[:, 1] = orig_h - corrs_rotated[:, 1]
    elif angle_option == 3:
        x, y = corrs_rotated[:, 0].copy(), corrs_rotated[:, 1].copy()
        corrs_rotated[:, 0] = orig_h - y
        corrs_rotated[:, 1] = x
    
    return plan_rotated, corrs_rotated

def load_image(path, corrs, is_photo):
    img = Image.open(path)

    # Handle GIFs by selecting the middle frame
    if path.lower().endswith(".gif"):
        img.seek(img.n_frames // 2)
    else:
        img_orientation = get_exif_orientation(img)
        corrs = transform_correspondences(corrs, img_orientation, img.size[0], img.size[1])
    img = exif_transpose(img).convert('RGB')
    return img, corrs

def load_images(pair, size, plan_corrs, photo_corrs, augment, verbose=False):
    plan_path, photo_path = pair
    image_views = []

    plan, plan_corrs = load_image(plan_path, plan_corrs, is_photo=False)
    photo, photo_corrs = load_image(photo_path, photo_corrs, is_photo=True)

    plan_W1, plan_H1 = plan.size
    photo_W1, photo_H1 = photo.size

    if augment:
        plan_augmented, plan_corrs = random_crop(plan, plan_corrs, crop_range=(0.95, 1.0))
        plan_augmented, plan_corrs = random_rotate(plan_augmented, plan_corrs)
        transform = ColorJitter
    else:
        plan_augmented = plan.copy()
        transform = ImgNorm
    plan_updated, plan_corrs_updated = resize_and_pad(plan_augmented, plan_corrs, size, is_photo=False)
    photo_updated, photo_corrs_updated = resize_and_pad(photo, photo_corrs, size, is_photo=True)

    plan_corrs_updated, photo_corrs_updated = crop_outlier_corrs(plan_corrs_updated, photo_corrs_updated, size, pair, plan_updated, photo_updated)

    plan_W2, plan_H2 = plan_updated.size
    photo_W2, photo_H2 = photo_updated.size

    if verbose:
        print(f' - adding {plan_path} with resolution {plan_W1}x{plan_H1} --> {plan_W2}x{plan_H2}')
        print(f' - adding {photo_path} with resolution {photo_W1}x{photo_H1} --> {photo_W2}x{photo_H2}')
    image_views.append(
        dict(
            img=transform(plan_updated)[None], 
            true_shape=np.int32([plan_updated.size[::-1]]), 
            idx=len(image_views), 
            instance=str(len(image_views)), 
            corrs=np.int32(plan_corrs_updated)
        )
    )
    image_views.append(
        dict(
            img=ImgNorm(photo_updated)[None], 
            true_shape=np.int32([photo_updated.size[::-1]]), 
            idx=len(image_views), 
            instance=str(len(image_views)), 
            corrs=np.int32(photo_corrs_updated)
        )
    )

    return image_views


if __name__ == "__main__":
    pass


