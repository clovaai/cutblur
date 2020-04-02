"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)

    # models
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--model", type=str, default="EDSR")

    # augmentations
    parser.add_argument("--use_moa", action="store_true")
    parser.add_argument("--augs", nargs="*", default=["none"])
    parser.add_argument("--prob", nargs="*", default=[1.0])
    parser.add_argument("--mix_p", nargs="*")
    parser.add_argument("--alpha", nargs="*", default=[1.0])
    parser.add_argument("--aux_prob", type=float, default=1.0)
    parser.add_argument("--aux_alpha", type=float, default=1.2)

    # dataset
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--dataset", type=str, default="DIV2K_SR")
    parser.add_argument("--camera", type=str, default="all") # RealSR
    parser.add_argument("--div2k_range", type=str, default="1-800/801-810")
    parser.add_argument("--scale", type=int, default=4) # SR
    parser.add_argument("--sigma", type=int, default=10) # DN
    parser.add_argument("--quality", type=int, default=10) # DeJPEG
    parser.add_argument("--type", type=int, default=1) # DeBlur

    # training setups
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=str, default="200-400-600")
    parser.add_argument("--gamma", type=int, default=0.5)
    parser.add_argument("--patch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=700000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gclip", type=int, default=0)

    # misc
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--ckpt_root", type=str, default="./pt")
    parser.add_argument("--save_root", type=str, default="./output")

    return parser.parse_args()


def make_template(opt):
    opt.strict_load = opt.test_only

    # model
    if "EDSR" in opt.model:
        opt.num_blocks = 32
        opt.num_channels = 256
        opt.res_scale = 0.1
    if "RCAN" in opt.model:
        opt.num_groups = 10
        opt.num_blocks = 20
        opt.num_channels = 64
        opt.reduction = 16
        opt.res_scale = 1.0
        opt.max_steps = 1000000
        opt.decay = "200-400-600-800"
        opt.gclip = 0.5 if opt.pretrain else opt.gclip
    if "CARN" in opt.model:
        opt.num_groups = 3
        opt.num_blocks = 3
        opt.num_channels = 64
        opt.res_scale = 1.0
        opt.batch_size = 64
        opt.decay = "400"

    # training setup
    if "DN" in opt.dataset or "JPEG" in opt.dataset:
        opt.max_steps = 1000000
        opt.decay = "300-550-800"
    if "RealSR" in opt.dataset:
        opt.patch_size *= opt.scale # identical (LR, HR) resolution

    # evaluation setup
    opt.crop = 6 if "DIV2K" in opt.dataset else 0
    opt.crop += opt.scale if "SR" in opt.dataset else 4

    # note: we tested on color DN task
    if "DIV2K" in opt.dataset or "DN" in opt.dataset:
        opt.eval_y_only = False
    else:
        opt.eval_y_only = True

    # default augmentation policies
    if opt.use_moa:
        opt.augs = ["blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
        opt.prob = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        opt.alpha = [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        opt.aux_prob, opt.aux_alpha = 1.0, 1.2
        opt.mix_p = None

        if "RealSR" in opt.dataset:
            opt.mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]

        if "DN" in opt.dataset or "JPEG" in opt.dataset:
            opt.prob = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        if "CARN" in opt.model and not "RealSR" in opt.dataset:
            opt.prob = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt
