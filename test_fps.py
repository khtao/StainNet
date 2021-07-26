import time

import numpy as np
import staintools
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from models import StainNet, ResnetGenerator

# read source image
img_source = Image.open("assets/3_color_net_neg23570_ori.png")
# read target image
img_target = Image.open("assets/3_color_net_neg23570_target.png")
# load  pretrained StainNet
model_Net = StainNet().cuda()
model_Net.load_state_dict(torch.load("checkpoints/StainNet/StainNet-3x0_best_psnr_layer3_ch32.pth"))
# load  pretrained StainGAN
model_GAN = ResnetGenerator(3, 3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=9).cuda()
model_GAN.load_state_dict(torch.load("checkpoints/StainGAN/latest_net_G_A.pth"))


def test_deeplearning_fps(model, n_iters, batchsize):
    data = torch.rand(batchsize, 3, 512, 512).cuda()
    start_time = time.time()
    for i in tqdm(range(n_iters)):
        with torch.no_grad():
            outputs = model(data)
    process_time = time.time() - start_time
    print("FPS is ", n_iters * batchsize / process_time)


def test_traditional_fps(source_img, ref_img, method, n_iters):
    ref_img = np.array(ref_img)
    source_img = np.array(ref_img)
    if method == "reinhard":
        normalizer = staintools.ReinhardColorNormalizer()
    else:
        normalizer = staintools.StainNormalizer(method=method)
    normalizer.fit(np.array(ref_img))
    start = time.time()
    for i in tqdm(range(n_iters)):
        try:
            img1_normalized = normalizer.transform(source_img)
        except:
            pass
    need_time = time.time() - start
    print(method, "FPS is ", n_iters / need_time)


print("test StainNet FPS")
test_deeplearning_fps(model_Net, 500, 100)
print("test StainGAN FPS")
test_deeplearning_fps(model_GAN, 100, 10)
test_traditional_fps(img_source, img_target, "reinhard", 200)
test_traditional_fps(img_source, img_target, "macenko", 50)
test_traditional_fps(img_source, img_target, "vahadane", 10)
