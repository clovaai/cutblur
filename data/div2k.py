"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import os
import glob
import data

class DIV2KSR(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root

        self.scale = opt.scale
        dir_HQ, dir_LQ = self.get_subdir()
        self.HQ_paths = sorted(glob.glob(os.path.join(root, dir_HQ, "*.png")))
        self.LQ_paths = sorted(glob.glob(os.path.join(root, dir_LQ, "*.png")))

        split = [int(n) for n in opt.div2k_range.replace("/", "-").split("-")]
        if phase == "train":
            s = slice(split[0]-1, split[1])
            self.HQ_paths, self.LQ_paths = self.HQ_paths[s], self.LQ_paths[s]
        else:
            s = slice(split[2]-1, split[3])
            self.HQ_paths, self.LQ_paths = self.HQ_paths[s], self.LQ_paths[s]

        super().__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = "DIV2K_train_HR"
        dir_LQ = "DIV2K_train_LR_bicubic/X{}".format(self.scale)
        return dir_HQ, dir_LQ


class DIV2KDN(DIV2KSR):
    def __init__(self, phase, opt):
        self.sigma = opt.sigma

        super().__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = "DIV2K_train_HR"
        dir_LQ = "DIV2K_train_DN/{}".format(self.sigma)
        return dir_HQ, dir_LQ


class DIV2KJPEG(DIV2KSR):
    def __init__(self, phase, opt):
        self.quality = opt.quality

        super().__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = "DIV2K_train_HR"
        dir_LQ = "DIV2K_train_JPEG/{}".format(self.quality)
        return dir_HQ, dir_LQ
