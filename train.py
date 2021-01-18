import argparse

import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import StainNet
from utils import ImageDataset
from utils import Visualizer
import os


def peak_signal_noise_ratio(image_true, image_test, data_range=None):
    err = torch.nn.MSELoss()(image_true, image_test)
    return 10 * torch.log10((data_range ** 2) / err)


def test(model, test_dataloader):
    total = len(test_dataloader)
    mean_metric = {}
    test_data = 0
    for i, (image, image_ori) in tqdm(enumerate(test_dataloader), total=total):
        image_ori = image_ori.cuda()
        image_ori = image_ori * 0.5 + 0.5
        image = image.cuda()
        test_data += int(image.size(0))
        for jj in range(2, 6):
            for ii in range(3, 8):
                key = str(jj) + str(ii)
                model[key].eval()
                if key not in mean_metric.keys():
                    mean_metric[key] = 0
                with torch.no_grad():
                    image_out = model[key](image)
                    image_out = image_out * 0.5 + 0.5
                    for ori, out in zip(image_ori, image_out):
                        ori = ori * 255.0
                        out = out * 255.0
                        psnr = peak_signal_noise_ratio(ori, out, data_range=255)
                        mean_metric[key] += psnr
    for jj in range(2, 6):
        for ii in range(3, 8):
            key = str(jj) + str(ii)
            mean_metric[key] = float(mean_metric[key] / test_data)
    return mean_metric


def train(opt):
    seq = iaa.Sequential([
        iaa.CropToFixedSize(opt.fineSize, opt.fineSize),
    ])
    dataset_train = ImageDataset(
        opt.source_root_train,
        opt.gt_root_train,
        transform=seq)
    dataset_test = ImageDataset(
        opt.source_root_test,
        opt.gt_root_test,
        transform=seq)
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batchSize,
                                  shuffle=True, num_workers=opt.nThreads)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchSize,
                                 shuffle=False, num_workers=opt.nThreads)
    models = {}
    optimizers = {}
    lrschedulrs = {}
    best_psnrs = {}
    for j in range(2, 6):
        for i in range(3, 8):
            models[str(j) + str(i)] = StainNet(opt.input_nc, opt.output_nc, j, 2 ** i, kernel_size=1)
            models[str(j) + str(i)] = nn.DataParallel(models[str(j) + str(i)]).cuda()
            optimizers[str(j) + str(i)] = SGD(models[str(j) + str(i)].parameters(), lr=opt.lr, momentum=0.9)
            lrschedulrs[str(j) + str(i)] = lr_scheduler.CosineAnnealingLR(optimizers[str(j) + str(i)], opt.epoch)
            best_psnrs[str(j) + str(i)] = 0
    loss_function = torch.nn.L1Loss()
    vis = Visualizer(env=opt.name)
    for i in range(opt.epoch):
        for j, (source_image, target_image) in tqdm(enumerate(dataloader_train)):
            target_image = target_image.cuda()
            source_image = source_image.cuda()
            for jj in range(2, 6):
                for ii in range(3, 8):
                    output = models[str(jj) + str(ii)](source_image)
                    loss = loss_function(output, target_image)
                    optimizers[str(jj) + str(ii)].zero_grad()
                    loss.backward()
                    optimizers[str(jj) + str(ii)].step()
            if (j + 1) % opt.display_freq == 0:
                vis.plot("loss", float(loss))
                vis.img("target image", target_image[0] * 0.5 + 0.5)
                vis.img("source image", source_image[0] * 0.5 + 0.5)
                vis.img("output", (output[0] * 0.5 + 0.5).clamp(0, 1))
        if (i + 1) % 1 == 0:
            test_result = test(models, dataloader_test)
            vis.plot_many(test_result)
            for jj in range(2, 6):
                for ii in range(3, 8):
                    if best_psnrs[str(jj) + str(ii)] < test_result[str(jj) + str(ii)]:
                        save_path = "{}/{}_best_psnr_layer{}_ch{}.pth".format(opt.checkpoints_dir, opt.name,
                                                                              jj, 2 ** ii)
                        best_psnrs[str(jj) + str(ii)] = test_result[str(jj) + str(ii)]
                        torch.save(models[str(jj) + str(ii)].module.state_dict(), save_path)
                        print(save_path, test_result)
        for jj in range(2, 6):
            for ii in range(3, 8):
                lrschedulrs[str(jj) + str(ii)].step()
                print("lrschedulr %s=" % (str(jj) + str(ii)), lrschedulrs[str(jj) + str(ii)].get_last_lr())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="StainNet-3x0", type=str,
                        help="name of the experiment.")
    parser.add_argument("--source_root_train", default="/media/khtao/My_Book/Dataset/StainNet_Dataset/train/source",
                        type=str,
                        help="path to source images for training")
    parser.add_argument("--gt_root_train", default="/media/khtao/My_Book/Dataset/StainNet_Dataset/train/StainGAN",
                        type=str,
                        help="path to ground truth images for training")
    parser.add_argument("--source_root_test", default="/media/khtao/My_Book/Dataset/StainNet_Dataset/test/source",
                        type=str,
                        help="path to source images for test")
    parser.add_argument("--gt_root_test", default="/media/khtao/My_Book/Dataset/StainNet_Dataset/test/StainGAN",
                        type=str,
                        help="path to ground truth images for test")
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--channels', type=int, default=32, help='# of channels in StainNet')
    parser.add_argument('--n_layer', type=int, default=4, help='# of layers in StainNet')
    parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
    parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--fineSize', type=int, default=256, help='crop to this size')
    parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
    parser.add_argument('--test_freq', type=int, default=5, help='frequency of cal')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for SGD')
    parser.add_argument('--epoch', type=int, default=200, help='how many epoch to train')
    parser.add_argument('--device_ids', type=str, default="0,1", help='how many epoch to train')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    train(opt=args)
