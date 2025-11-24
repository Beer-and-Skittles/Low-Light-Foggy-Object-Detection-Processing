import numpy as np
from PIL import Image
from skimage import exposure, filters, color

def to_np(pil_img):
    return np.array(pil_img.convert("RGB"))

def to_pil(np_img):
    return Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))

def clahe(img_np):
    img_float = img_np / 255.0
    lab = color.rgb2lab(img_float)
    lab[..., 0] = exposure.equalize_adapthist(lab[..., 0] / 100.0) * 100.0
    return color.lab2rgb(lab) * 255.0

def gamma(img_np, g=0.6):
    return (img_np / 255.0) ** g * 255.0

def retinex(img_np, sigma=80):
    img_float = img_np / 255.0
    blurred = np.zeros_like(img_float)
    for c in range(3):
        blurred[..., c] = filters.gaussian(img_float[..., c], sigma=sigma)
    ret = np.log(img_float + 1e-6) - np.log(blurred + 1e-6)
    ret = (ret - ret.min()) / (ret.max() - ret.max() + 1e-6)
    return ret * 255.0

def hist_match(img_np, ref_np):
    img_float = img_np / 255.0
    ref_float = ref_np / 255.0
    matched = exposure.match_histograms(img_float, ref_float, channel_axis=-1)
    return matched * 255.0
