import time

import numpy as np
import openslide
import torch
import torch.nn as nn
from models import StainNet, ResnetGenerator
import staintools
from tqdm import tqdm


def read_whole_slide(image_path, patchsize=512):
    whole_slide = []
    image_slide = openslide.open_slide(image_path)
    width, height = image_slide.dimensions
    # width, height = 5120, 5120
    print(image_slide.dimensions)
    for x in range(0, width, patchsize):
        for y in range(0, height, patchsize):
            whole_slide.append(np.array(image_slide.read_region((x, y), 0, (patchsize, patchsize)).convert("RGB")))
    return whole_slide


def test_whole_slide(model, data, factor=1):
    batchsize = 20
    for i in range(len(data) // batchsize):
        imgs = np.stack(data[i * batchsize:i * batchsize + batchsize])
        imgs = imgs.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).cuda()
            imgs = (imgs - 0.5) * 2
    start_time = time.time()
    for i in range(len(data) // batchsize):
        imgs = np.stack(data[i * batchsize:i * batchsize + batchsize])
        imgs = imgs.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).cuda()
            imgs = (imgs - 0.5) * 2
    io_time = time.time()
    print("io_time:", (io_time - start_time) / factor, "s")
    io_need = io_time - start_time
    for i in range(len(data) // batchsize):
        imgs = np.stack(data[i * batchsize:i * batchsize + batchsize])
        imgs = imgs.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).cuda()
            imgs = (imgs - 0.5) * 2
            outputs = model(imgs)
    process_time = time.time() - io_time - io_need
    print("whole slide time is ", process_time / factor, "s")
    print("FPS is ", len(data) / process_time)


def test_traditional_whole_slide(target_img, method, data):
    if method == "reinhard":
        normalizer = staintools.ReinhardColorNormalizer()
    else:
        normalizer = staintools.StainNormalizer(method=method)
    target = staintools.read_image(target_img)
    normalizer.fit(target)
    start = time.time()
    for img1 in tqdm(data, total=len(data)):
        try:
            img1_normalized = normalizer.transform(img1)
        except:
            pass
    need_time = time.time() - start
    print(method, "whole slide time is ", need_time, "s")
    print(method, "FPS is ", len(data) / need_time)


def test_time_cytopathology():
    image_path = "test_slides/Cytopathology/1162026.svs"
    data = read_whole_slide(image_path)
    model_path = "checkpoints/stainnet/Cytopathology/stainnet_staingan_best_psnr_layer3_ch32.pth"
    model = StainNet(3, 32).cuda()
    model.load_state_dict(torch.load(model_path))
    print(model_path)
    test_whole_slide(model, data)
    model_path = "checkpoints/staingan/Cytopathology/latest_net_G_A.pth"
    model = ResnetGenerator(3, 3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=9).cuda()
    model.load_state_dict(torch.load(model_path))
    print(model_path)
    test_whole_slide(model, data)
    target_img = "test_slides/Cytopathology/target.jpg"
    test_traditional_whole_slide(target_img, "reinhard", data)
    test_traditional_whole_slide(target_img, "macenko", data)
    test_traditional_whole_slide(target_img, "vahadane", data)


def test_time_histopathology():
    image_path = "test_slides/Histopathology/test_001.tif"
    data = read_whole_slide(image_path, 256)
    model_path = "checkpoints/stainnet/Histopathology/stainnet_staingan_hist_best_psnr_layer3_ch32.pth"
    model = StainNet(3, 3, 3, 32).cuda()
    model.load_state_dict(torch.load(model_path))
    print(model_path)
    test_whole_slide(model, data)
    model_path = "checkpoints/staingan/Histopathology/latest_net_G_A.pth"
    model = ResnetGenerator(3, 3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=9).cuda()
    model.load_state_dict(torch.load(model_path))
    print(model_path)
    test_whole_slide(model, data)
    target_img = "test_slides/Histopathology/H03_00A0_3.jpg"
    test_traditional_whole_slide(target_img, "reinhard", data)
    test_traditional_whole_slide(target_img, "macenko", data)
    test_traditional_whole_slide(target_img, "vahadane", data)


if __name__ == '__main__':
    test_time_histopathology()
    test_time_cytopathology()
