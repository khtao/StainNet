import argparse
import os
import random
from datetime import datetime

import cv2
import imageio
import numpy as np
import staintools
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import DataLoader
import shutil
from models import StainNet, ResnetGenerator
from utils import list_file_tree, SingleImage
from tqdm import tqdm
import subprocess


def detect_image(opt, model):
    dataset = SingleImage(opt.source_dir)
    dataloader = DataLoader(dataset,
                            batch_size=10,
                            num_workers=4,
                            drop_last=False)
    file_list = dataset.image_list
    save_path = os.path.join(opt.save_root, os.path.split(opt.model_path)[1][:-4])
    os.makedirs(save_path, exist_ok=True)
    num = 0
    for imgs in tqdm(dataloader):
        with torch.no_grad():
            imgs = imgs.cuda()
            imgs = (imgs - 0.5) * 2
            outputs = model(imgs)
            outputs = outputs * 0.5 + 0.5
            outputs = outputs.clamp(0, 1).detach().cpu().numpy()
        for out in outputs:
            file_path = file_list[num]
            file_path = os.path.join(save_path, os.path.split(file_path)[1])
            imageio.imwrite(file_path[:-4] + ".png",
                            (out * 255).astype(np.uint8).transpose((1, 2, 0)))
            num += 1
    return save_path


def traditional_methods(opt):
    image_source = list_file_tree(opt.source_dir, "png")
    image_target = list_file_tree(opt.gt_dir, "png")
    image_source.sort()
    image_target.sort()
    if opt.method == "reinhard":
        normalizer = staintools.ReinhardColorNormalizer()
    else:
        normalizer = staintools.StainNormalizer(method=opt.method)

    if opt.random_target:
        num = random.randint(0, len(image_target) - 1)
        save_path = os.path.join(opt.save_root, opt.method + "_random")
        os.makedirs(save_path, exist_ok=True)
        print("target choose:", image_target[num])
        target = staintools.read_image(image_target[num])
        normalizer.fit(target)
        for source in tqdm(image_source):
            img = staintools.read_image(source)
            filename = os.path.split(source)[1]
            try:
                img_normalized = normalizer.transform(img)
                imageio.imwrite(os.path.join(save_path, filename[:-4] + ".png"),
                                img_normalized)
            except:
                print("error in ", source)
    else:
        save_path = os.path.join(opt.save_root, opt.method + "_matched")
        os.makedirs(save_path, exist_ok=True)
        for source, target in tqdm(zip(image_source, image_target)):
            img1 = staintools.read_image(source)
            img2 = staintools.read_image(target)
            filename = os.path.split(source)[1]
            try:
                normalizer.fit(img2)
                img1_normalized = normalizer.transform(img1)
                imageio.imwrite(os.path.join(save_path, filename[:-4] + ".png"),
                                img1_normalized)
            except:
                print("error in ", source)
    return save_path


def test_result(opt):
    print(opt.result_dir, opt.source_dir, opt.gt_dir)
    reslut_files = list_file_tree(opt.result_dir, "png")
    source_files = list_file_tree(opt.source_dir, "png")
    target_files = list_file_tree(opt.gt_dir, "png")
    reslut_files.sort()
    source_files.sort()
    target_files.sort()
    all_metirc = []
    for reslut, source, target in tqdm(zip(reslut_files, source_files, target_files)):
        image0 = imageio.imread(reslut)
        image1 = imageio.imread(source)
        image2 = imageio.imread(target)
        ssim = structural_similarity(image0, image2, win_size=11, gaussian_weights=True, multichannel=True,
                                     K1=0.01,
                                     K2=0.03,
                                     sigma=1.5, data_range=255)
        psnr = peak_signal_noise_ratio(image0, image2, data_range=255)
        image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image0 = image0.astype(np.float)
        image1 = image1.astype(np.float)
        image0 = (image0 - image0.min()) / (image0.max() - image0.min()) * 255
        image1 = (image1 - image1.min()) / (image1.max() - image1.min()) * 255
        ssim_source = structural_similarity(image0, image1, win_size=11, gaussian_weights=True, K1=0.01,
                                            K2=0.03,
                                            sigma=1.5, data_range=255)
        all_metirc.append([source, {"ssim": ssim, "psnr": psnr, "ssim_source": ssim_source}])

    mean_ssim = sum([k[1]["ssim"] for k in all_metirc]) / len(reslut_files)
    mean_psnr = sum([k[1]["psnr"] for k in all_metirc]) / len(reslut_files)
    mean_ssim_source = sum([k[1]["ssim_source"] for k in all_metirc]) / len(reslut_files)
    torch.save(all_metirc, os.path.join(opt.result_dir, "all_metirc.data"))
    print("SSIM GT", mean_ssim,
          "PSNR GT", mean_psnr,
          "SSIM Source", mean_ssim_source)
    return mean_ssim, mean_psnr, mean_ssim_source


def khan(source, target):
    shutil.copy(source, "source.jpg")
    fs = open("khan/filename.txt", "w")
    fs.write("source.jpg" + "\n")
    fs.close()
    subprocess.run("wine khan/ColourNormalisation.exe BimodalDeconvRVM "
                   "khan/filename.txt " + target + " khan/HE.colourmodel",
                   shell=True)
    return imageio.imread("normalised/source.jpg")


def khan_method(opt):
    image_source = list_file_tree(opt.source_dir, "png")
    image_target = list_file_tree(opt.gt_dir, "png")
    image_source.sort()
    image_target.sort()
    os.makedirs("normalised", exist_ok=True)
    if opt.random_target:
        num = random.randint(0, len(image_target) - 1)
        save_path = os.path.join(opt.save_root, opt.method + "_random")
        os.makedirs(save_path, exist_ok=True)
        print("target choose:", image_target[num])
        for source in image_source:
            image = khan(source, image_target[num])
            filename = os.path.split(source)[1]
            imageio.imwrite(os.path.join(save_path, filename[:-4] + ".png"), image)

    else:
        save_path = os.path.join(opt.save_root, opt.method + "_matched")
        os.makedirs(save_path, exist_ok=True)
        for source, target in zip(image_source, image_target):
            image = khan(source, target)
            filename = os.path.split(source)[1]
            imageio.imwrite(os.path.join(save_path, filename[:-4] + ".png"), image)

    return save_path


def test_methods(opt):
    opt.save_root = os.path.split(opt.source_dir)[0]
    if opt.method == "StainNet":
        model = StainNet(opt.input_nc, opt.output_nc, opt.n_layer, opt.channels, kernel_size=7)
        model = model.cuda()
        model.load_state_dict(torch.load(opt.model_path))
        model.eval()
        opt.result_dir = detect_image(opt, model)
    elif opt.method == "StainGAN":
        model = ResnetGenerator(opt.input_nc, opt.output_nc, ngf=64, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9)
        model = model.cuda()
        model.load_state_dict(torch.load(opt.model_path))
        model.eval()
        opt.result_dir = detect_image(opt, model)
    elif opt.method in ["reinhard", "macenko", "vahadane"]:
        opt.result_dir = traditional_methods(opt)
    elif opt.method == "khan":
        opt.result_dir = khan_method(opt)
    else:
        raise RuntimeError("Not implemented Error!")
    print("result save to ", opt.result_dir)
    if opt.test_ssim:
        mean_ssim, mean_psnr, mean_ssim_source = test_result(opt)
        fs = open(os.path.join(opt.result_dir, "result.txt"), "a+")
        fs.write(
            "{}, SSIM GT:{}, PSNR GT:{}, SSIM Source:{}\n".format(datetime.now(), mean_ssim, mean_psnr,
                                                                  mean_ssim_source))
        print("test result save to ", os.path.join(opt.result_dir, "result.txt"))


def test_models(opt):
    opt.save_root = os.path.split(opt.source_dir)[0]
    # for j in range(2, 6):
    #     for i in range(3, 9):
    model = StainNet(opt.input_nc, opt.output_nc, 3, 32, kernel_size=1)
    model = model.cuda()
    opt.model_path = "/home/khtao/workplace/MEGA/研究生学习/StainNet/checkpoints/" \
                     "stainnet_new/StainNet-3x0_best_psnr_layer3_ch32.pth"
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    opt.result_dir = detect_image(opt, model)
    # mean_ssim, mean_psnr, mean_ssim_source = test_result(opt)
    # fs = open(os.path.join(opt.result_dir, "result.txt"), "a+")
    # fs.write(
    #     "{}, SSIM GT:{}, PSNR GT:{}, SSIM Source:{}\n".format(datetime.now(), mean_ssim, mean_psnr,
    #                                                           mean_ssim_source))
    # print("test result save to ", os.path.join(opt.result_dir, "result.txt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str,
                        help="path to result images for test")
    parser.add_argument("--result_dir", type=str,
                        help="path to result images for test")
    parser.add_argument("--source_dir", default="/home/khtao/kkk", type=str,
                        help="path to source images for test")
    parser.add_argument("--gt_dir", default="/home/khtao/视频/tttt/target", type=str,
                        help="path to ground truth images for test")
    parser.add_argument("--method", default="StainNet", type=str,
                        help="different methods for test must be one of { StainNet StainGAN reinhard macenko vahadane khan }")
    parser.add_argument('--test_ssim', action="store_true", default=True,
                        help='whether calculate SSIM , default is False')
    parser.add_argument('--random_target', action="store_true", default=False,
                        help='random choose target or using matched ground truth, True is random choose target')

    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--channels', type=int, default=32, help='# of channels in StainNet')
    parser.add_argument('--n_layer', type=int, default=3, help='# of layers in StainNet')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/StainNet-7x7_best_psnr_layer3_ch32.pth',
                        help='models path to load')

    args = parser.parse_args()
    # print(args)
    # print("args.test_ssim", args.test_ssim)
    test_models(opt=args)
