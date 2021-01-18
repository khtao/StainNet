#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import staintools
import torch
import os
import torch.nn as nn
from tqdm import tqdm
from models import StainNet, ResnetGenerator
from PIL import Image
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# In[2]:


# read source image
img_source = Image.open("assets/3_color_net_neg23570_ori.png")
plt.imshow(img_source)

# In[3]:


# read target image
img_target = Image.open("assets/3_color_net_neg23570_target.png")
plt.imshow(img_target)

# In[4]:


# load  pretrained StainNet
model_Net = StainNet().cuda()
model_Net.load_state_dict(torch.load("checkpoints/StainNet/StainNet-3x0_best_psnr_layer3_ch32.pth"))


# In[5]:


def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image = image[np.newaxis, ...]
    image = torch.from_numpy(image)
    return image


def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1, 2, 0))
    return image


# In[6]:


# run normlization
image_net = model_Net(norm(img_source).cuda())
image_net = un_norm(image_net)
plt.imshow(image_net)

# In[7]:


# load  pretrained StainGAN
model_GAN = ResnetGenerator(3, 3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=9).cuda().cuda()
model_GAN.load_state_dict(torch.load("checkpoints/StainGAN/latest_net_G_A.pth"))

# In[8]:


# run normlization
image_gan = model_GAN(norm(img_source).cuda())
image_gan = un_norm(image_gan)
plt.imshow(image_gan)

# In[9]:


# run reinhard normlization
print("reinhard")
normalizer = staintools.ReinhardColorNormalizer()
normalizer.fit(np.array(img_target))
reinhard_normalized = normalizer.transform(np.array(img_source))
plt.imshow(reinhard_normalized)

# In[10]:


# run macenko normlization
normalizer = staintools.StainNormalizer(method="macenko")
normalizer.fit(np.array(img_target))
macenko_normalized = normalizer.transform(np.array(img_source))
plt.imshow(macenko_normalized)

# In[11]:


# run vahadane normlization
normalizer = staintools.StainNormalizer(method="vahadane")
normalizer.fit(np.array(img_target))
vahadane_normalized = normalizer.transform(np.array(img_source))
plt.imshow(vahadane_normalized)


# In[12]:


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


# In[ ]:


print("test StainNet FPS")
test_deeplearning_fps(model_Net, 500, 100)

# In[14]:


print("test StainGAN FPS")
test_deeplearning_fps(model_GAN, 100, 10)

# In[17]:


test_traditional_fps(img_source, img_target, "reinhard", 200)
test_traditional_fps(img_source, img_target, "macenko", 50)
test_traditional_fps(img_source, img_target, "vahadane", 10)

# In[ ]:


# In[ ]:



