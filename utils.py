import os
import time
from glob import glob

import lmdb
import numpy as np
import visdom
from PIL import Image
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import dataset
from tqdm import tqdm


def list_file_tree(path, file_type="tif"):
    if file_type.find("*") < 0:
        file_type = "*" + file_type
    image_list = glob(os.path.join(path, "*" + file_type), recursive=True)
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


class ImageClassDataset(dataset.Dataset):
    def __init__(self, pos_path, neg_path, use_lmdb=False, augment=None, transform=None):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.augment = augment
        self.use_lmdb = use_lmdb
        self.transform = transform
        self.pos_list = list_file_tree(pos_path, "png")
        self.neg_list = list_file_tree(neg_path, "png")
        self.image_list = self.pos_list + self.neg_list
        if self.use_lmdb:
            self.lmdb = self.make_lmdb(os.path.join(self.pos_path, "lmdb"))

    def __len__(self):
        return len(self.image_list)

    def make_lmdb(self, path):
        length = len(self.image_list)
        if os.path.exists(path):
            env = lmdb.open(path, map_size=10737418240)
            txn = env.begin()
            num = txn.get("len".encode())
            if num is None or int(txn.get("len".encode())) != length:
                os.remove(path + "/data.mdb")
                os.remove(path + "/lock.mdb")
            else:
                return txn
        env = lmdb.open(path, map_size=10737418240)
        txn = env.begin(write=True)
        for idx in tqdm(range(length)):
            image = Image.open(self.image_list[idx]).convert("RGB").resize((256, 256))
            label = 1 if idx < len(self.pos_list) else 0
            label = str(label)
            buff = cv2.imencode(".png", np.array(image, dtype=np.uint8))[1]
            txn.put(key=("image" + str(idx)).encode(), value=buff.tobytes())
            txn.put(key=("label" + str(idx)).encode(), value=label.encode())
        txn.put(key="len".encode(), value=str(length).encode())
        txn.commit()
        return env.begin()

    def __getitem__(self, item):
        if self.use_lmdb:
            image = self.lmdb.get(key=("image" + str(item)).encode())
            label = self.lmdb.get(key=("label" + str(item)).encode())
            label = int(label.decode())
            image = np.frombuffer(image, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = Image.fromarray(image)
        else:
            image = Image.open(self.image_list[item]).convert("RGB").resize((256, 256))
            label = 1 if item < len(self.pos_list) else 0
        if self.augment:
            image = self.augment(image)
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)
        image = (np.array(image, dtype=np.float32) - 128).transpose((2, 0, 1)) / 128.0
        return image, label


class ImageDataset(dataset.Dataset):
    def __init__(self, data_path, seg_path, transform=None, augment=None):
        self.data_path = data_path
        self.seg_path = seg_path
        self.transform = transform
        self.augment = augment
        self.image_list = list_file_tree(os.path.join(data_path), "png")
        self.image_list += list_file_tree(os.path.join(data_path), "jpg")
        self.seg_list = list_file_tree(os.path.join(seg_path), "png")
        self.seg_list += list_file_tree(os.path.join(seg_path), "jpg")

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
        self.image_list = list_file_tree(os.path.join(data_path), "png")
        self.image_list += list_file_tree(os.path.join(data_path), "jpg")
        # assert len(self.image_list) == len(self.cyt_list)
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img = Image.open(self.image_list[item])
        img = (np.array(img, dtype=np.float32) / 255.0).transpose((2, 0, 1))
        return img


class incre_std_avg():
    '''
    增量计算海量数据平均值和标准差,方差
    1.数据
    obj.avg为平均值
    obj.std为标准差
    obj.n为数据个数
    对象初始化时需要指定历史平均值,历史标准差和历史数据个数(初始数据集为空则可不填写)
    2.方法
    obj.incre_in_list()方法传入一个待计算的数据list,进行增量计算,获得新的avg,std和n(海量数据请循环使用该方法)
    obj.incre_in_value()方法传入一个待计算的新数据,进行增量计算,获得新的avg,std和n(海量数据请将每个新参数循环带入该方法)
    '''

    def __init__(self, h_avg=0, h_std=0, n=0):
        self.avg = h_avg
        self.std = h_std
        self.n = n

    def incre_in_list(self, new_list):
        avg_new = np.mean(new_list, dtype=np.longdouble)
        incre_avg = (self.n * self.avg + len(new_list) * avg_new) / \
                    (self.n + len(new_list))
        std_new = np.std(new_list, dtype=np.longdouble)
        incre_std = np.sqrt((self.n * (self.std ** 2 + (incre_avg - self.avg) ** 2) + len(new_list)
                             * (std_new ** 2 + (incre_avg - avg_new) ** 2)) / (self.n + len(new_list)),
                            dtype=np.longdouble)
        self.avg = incre_avg
        self.std = incre_std
        self.n += len(new_list)

    def incre_in_value(self, value):
        incre_avg = (self.n * self.avg + value) / (self.n + 1)
        incre_std = np.sqrt((self.n * (self.std ** 2 + (incre_avg - self.avg)
                                       ** 2) + (incre_avg - value) ** 2) / (self.n + 1), dtype=np.longdouble)
        self.avg = incre_avg
        self.std = incre_std
        self.n += 1

    def incre_in_std_mean(self, num, mean, std):
        incre_avg = (self.n * self.avg + num * mean) / (self.n + num)
        incre_std = np.sqrt((self.n * (self.std ** 2 + (incre_avg - self.avg) ** 2) + num
                             * (std ** 2 + (incre_avg - mean) ** 2)) / (self.n + num),
                            dtype=np.longdouble)
        self.avg = incre_avg
        self.std = incre_std
        self.n += num


if __name__ == '__main__':
    import shutil
    import torch
    from datetime import datetime

    files = list_file_tree("/media/khtao/My_Book/Dataset/StainNet_Dataset/test/source", "png")
    for tt in ["StainNet", "StainGAN", "reinhard_random", "reinhard_matched", "macenko_random", "macenko_matched",
               "vahadane_matched", "vahadane_random"]:
        target = "/home/khtao/data/colornet/color_net_new/" + tt
        save_path = "/media/khtao/My_Book/Dataset/StainNet_Dataset/test/" + tt
        os.makedirs(save_path, exist_ok=True)
        all_metirc = torch.load(os.path.join(target, "all_metirc.data"))
        all_metirc_files = [os.path.split(k[0])[1] for k in all_metirc]
        all_metirc_new = []
        for file in files:
            filename = os.path.split(file)[1]
            shutil.copy(os.path.join(target, filename),
                        os.path.join(save_path, filename))
            k = all_metirc_files.index(filename)
            all_metirc_new.append(all_metirc[k])
            print(filename, all_metirc[k][0])
        mean_ssim = sum([k[1]["ssim"] for k in all_metirc_new]) / len(all_metirc_new)
        mean_psnr = sum([k[1]["psnr"] for k in all_metirc_new]) / len(all_metirc_new)
        mean_ssim_source = sum([k[1]["ssim_source"] for k in all_metirc_new]) / len(all_metirc_new)
        print(tt, mean_ssim, mean_psnr, mean_ssim_source)
        torch.save(all_metirc, os.path.join(save_path, "all_metirc.data"))
        fs = open(os.path.join(save_path, "result.txt"), "a+")
        fs.write(
            "{}, SSIM GT:{}, PSNR GT:{}, SSIM Source:{}\n".format(datetime.now(), mean_ssim, mean_psnr,
                                                                  mean_ssim_source))

    #
    # np.random.shuffle(files)
    # total = len(files)
    # test_num = int(total * 0.3)
    # train_num = total - test_num
    # for file in files[:test_num]:
    #     filename = os.path.split(file)[1]
    #     shutil.copy(file, os.path.join(save_path, "test", "source", filename))
    #     shutil.copy(file.replace("dataA", "dataB"), os.path.join(save_path, "test", "target", filename))
    #
    # for file in files[test_num:]:
    #     filename = os.path.split(file)[1]
    #     shutil.copy(file, os.path.join(save_path, "train", "source", filename))
    #     shutil.copy(file.replace("dataA", "dataB"), os.path.join(save_path, "train", "target", filename))
