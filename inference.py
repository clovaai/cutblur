"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import os
import glob
import importlib
from tqdm import tqdm
import numpy as np
import skimage.io as io
import skimage.color as color
import torch
import torch.nn.functional as F
import option

def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor


@torch.no_grad()
def main(opt):
    os.makedirs(opt.save_root, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(opt.model.lower()))
    net = module.Net(opt).to(dev)

    state_dict = torch.load(opt.pretrain, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)

    paths = sorted(glob.glob(os.path.join(opt.dataset_root, "*.png")))
    for path in tqdm(paths):
        name = path.split("/")[-1]

        LR = color.gray2rgb(io.imread(path))
        LR = im2tensor(LR).unsqueeze(0).to(dev)
        LR = F.interpolate(LR, scale_factor=opt.scale, mode="nearest")

        SR = net(LR).detach()
        SR = SR[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()

        save_path = os.path.join(opt.save_root, name)
        io.imsave(save_path, SR)


if __name__ == "__main__":
    opt = option.get_option()
    main(opt)
