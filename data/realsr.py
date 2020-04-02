"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import os
import glob
import data

class RealSR(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root

        self.scale = opt.scale

        subdir = "Train" if phase == "train" else "Test"
        path_canon_all = sorted(glob.glob(os.path.join(
            root, "Canon", subdir, str(self.scale), "*.png"
        )))
        path_nikon_all = sorted(glob.glob(os.path.join(
            root, "Nikon", subdir, str(self.scale), "*.png"
        )))

        path_canon_HR = [p for p in path_canon_all if "HR" in p]
        path_canon_LR = [p for p in path_canon_all if "LR" in p]
        path_nikon_HR = [p for p in path_nikon_all if "HR" in p]
        path_nikon_LR = [p for p in path_nikon_all if "LR" in p]

        if opt.camera == "canon":
            self.HQ_paths = path_canon_HR
            self.LQ_paths = path_canon_LR
        elif opt.camera == "nikon":
            self.HQ_paths = path_nikon_HR
            self.LQ_paths = path_nikon_LR
        elif opt.camera == "all":
            self.HQ_paths = path_canon_HR+path_nikon_HR
            self.LQ_paths = path_canon_LR+path_nikon_LR
        else:
            raise ValueError("camera must be one of the [canon, nikon, all].")

        super().__init__(phase, opt)
