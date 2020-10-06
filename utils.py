import os
from glob import glob
import time
import numpy as np
import visdom
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from PIL import Image
from torch.utils.data import dataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def list_file_tree(path, file_type="tif"):
    image_list = list()
    dir_list = os.listdir(path)
    if os.path.isdir(path):
        image_list += glob(os.path.join(path, "*" + file_type))
    for dir_name in dir_list:
        sub_path = os.path.join(path, dir_name)
        if os.path.isdir(sub_path):
            image_list += list_file_tree(sub_path, file_type)
    return image_list


class Visualizer(object):
    """
    wrapper for visdom
    you can still access naive visdom function by
    self.line, self.scater,self._send,etc.
    due to the implementation of `__getattr__`
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self._vis_kw = kwargs

        # e.g.（’loss',23） the 23th value of loss
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        change the config of visdom
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(
            env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self


def compute_psnr_and_ssim(image1, image2, border_size=0):
    """
    Computes PSNR and SSIM index from 2 images.
    We round it and clip to 0 - 255. Then shave 'scale' pixels from each border.
    """
    if border_size > 0:
        image1 = image1[border_size:-border_size, border_size:-border_size, :]
        image2 = image2[border_size:-border_size, border_size:-border_size, :]

    psnr = peak_signal_noise_ratio(image1, image2, data_range=255)
    ssim = structural_similarity(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01,
                                 K2=0.03,
                                 sigma=1.5, data_range=255)
    return psnr, ssim


class ImageDataset(dataset.Dataset):
    def __init__(self, data_path, seg_path, transform=None, augment=None):
        self.data_path = data_path
        self.seg_path = seg_path
        self.transform = transform
        self.augment = augment
        self.image_list = list_file_tree(os.path.join(data_path), "jpg")
        self.seg_list = list_file_tree(os.path.join(seg_path), "png")

        assert len(self.image_list) == len(self.seg_list)
        self.image_list.sort()
        self.seg_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = Image.open(self.image_list[item]).convert("RGB")
        image_ori = Image.open(self.seg_list[item]).convert("RGB")
        if self.augment:
            image = self.augment(image)
        image_ori = np.array(image_ori)
        image = np.array(image)
        image_ori = SegmentationMapsOnImage(image_ori, shape=image_ori.shape)
        if self.transform:
            image, image_ori = self.transform(image=image, segmentation_maps=image_ori)
        image_ori = image_ori.get_arr()
        image = (np.array(image, dtype=np.float32) / 255.0).transpose((2, 0, 1))
        image_ori = (np.array(image_ori, dtype=np.float32) / 255.0).transpose((2, 0, 1))
        image = (image - 0.5) * 2
        image_ori = (image_ori - 0.5) * 2
        return image, image_ori


class SingleImage(dataset.Dataset):
    def __init__(self, data_path, transform=None, augment=None):
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        self.image_list = list_file_tree(os.path.join(data_path), "jpg")
        # assert len(self.image_list) == len(self.cyt_list)
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img = Image.open(self.image_list[item])
        img = (np.array(img, dtype=np.float32) / 255.0).transpose((2, 0, 1))
        return img
