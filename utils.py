"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import random
import numpy as np

def crop(HQ, LQ, psize, scale=4):
    h, w = LQ.shape[:-1]
    x = random.randrange(0, w-psize+1)
    y = random.randrange(0, h-psize+1)

    crop_HQ = HQ[y*scale:y*scale+psize*scale, x*scale:x*scale+psize*scale]
    crop_LQ = LQ[y:y+psize, x:x+psize]

    return crop_HQ.copy(), crop_LQ.copy()


def flip_and_rotate(HQ, LQ):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    if hflip:
        HQ, LQ = HQ[:, ::-1, :], LQ[:, ::-1, :]
    if vflip:
        HQ, LQ = HQ[::-1, :, :], LQ[::-1, :, :]
    if rot90:
        HQ, LQ = HQ.transpose(1, 0, 2), LQ.transpose(1, 0, 2)

    return HQ, LQ


def rgb2ycbcr(img, y_only=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.

    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
            [24.966, 112.0, -18.214]]
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))
