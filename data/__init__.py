"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import os
import glob
import importlib
import numpy as np
import skimage.io as io
import skimage.color as color
import torch
import utils

def generate_loader(phase, opt):
    cname = opt.dataset.replace("_", "")
    if "DIV2K" in opt.dataset:
        mname = importlib.import_module("data.div2k")
    elif "RealSR" in opt.dataset:
        mname = importlib.import_module("data.realsr")
    elif "SR" in opt.dataset: # SR benchmark datasets
        mname = importlib.import_module("data.benchmark")
        cname = "BenchmarkSR"
    elif "DN" in opt.dataset: # DN benchmark datasets
        mname = importlib.import_module("data.benchmark")
        cname = "BenchmarkSR"
    elif "JPEG" in opt.dataset: # JPEG benchmark datasets
        mname = importlib.import_module("data.benchmark")
        cname = "BenchmarkSR"
    else:
        raise ValueError("Unsupported dataset: {}".format(opt.dataset))

    kwargs = {
        "batch_size": opt.batch_size if phase == "train" else 1,
        "num_workers": opt.num_workers if phase == "train" else 0,
        "shuffle": phase == "train",
        "drop_last": phase == "train",
    }

    dataset = getattr(mname, cname)(phase, opt)
    return torch.utils.data.DataLoader(dataset, **kwargs)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt):
        print("Load dataset... (phase: {}, len: {})".format(phase, len(self.HQ_paths)))
        self.HQ, self.LQ = list(), list()
        for HQ_path, LQ_path in zip(self.HQ_paths, self.LQ_paths):
            self.HQ += [io.imread(HQ_path)]
            self.LQ += [io.imread(LQ_path)]

        self.phase = phase
        self.opt = opt

    def __getitem__(self, index):
        # follow the setup of EDSR-pytorch
        if self.phase == "train":
            index = index % len(self.HQ)

        def im2tensor(im):
            np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_t).float()
            return tensor

        HQ, LQ = self.HQ[index], self.LQ[index]
        if len(HQ.shape) < 3:
            HQ = color.gray2rgb(HQ)
        if len(LQ.shape) < 3:
            LQ = color.gray2rgb(LQ)

        if self.phase == "train":
            inp_scale = HQ.shape[0] // LQ.shape[0]
            HQ, LQ = utils.crop(HQ, LQ, self.opt.patch_size, inp_scale)
            HQ, LQ = utils.flip_and_rotate(HQ, LQ)
        return im2tensor(HQ), im2tensor(LQ)

    def __len__(self):
        # follow the setup of EDSR-pytorch
        if self.phase == "train":
            return (1000 * self.opt.batch_size) // len(self.HQ) * len(self.HQ)
        return len(self.HQ)
