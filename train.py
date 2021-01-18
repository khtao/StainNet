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
from utils import compute_psnr_and_ssim


def test(model, test_dataloader):
    model.eval()
    total = len(test_dataloader)
    mean_psnr, mean_ssim = 0.0, 0.0
    test_data = 0
    for i, (image, image_ori) in tqdm(enumerate(test_dataloader), total=total):
        with torch.no_grad():
            image_out = model(image.cuda())
            image_out = image_out * 0.5 + 0.5
            image_ori = image_ori * 0.5 + 0.5
        for ori, out in zip(image_ori, image_out):
            ori = ori.detach().cpu().numpy() * 255.0
            ori = ori.transpose((1, 2, 0))

            out = out.detach().cpu().numpy() * 255.0
            out = out.transpose((1, 2, 0))

            psnr, ssim = compute_psnr_and_ssim(ori.astype(np.uint8),
                                               out.astype(np.uint8), border_size=2)
            mean_psnr += psnr
            mean_ssim += ssim
            test_data += 1

    mean_ssim /= test_data
    mean_psnr /= test_data
    return {"psnr": mean_psnr, "ssim": mean_ssim}


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
    model = StainNet(opt.input_nc, opt.output_nc, opt.n_layer, opt.channels)
    model = nn.DataParallel(model).cuda()
    optimizer = SGD(model.parameters(), lr=opt.lr)
    loss_function = torch.nn.L1Loss()
    lrschedulr = lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch)
    vis = Visualizer(env=opt.name)
    best_psnr = 0
    for i in range(opt.epoch):
        for j, (source_image, target_image) in tqdm(enumerate(dataloader_train)):
            target_image = target_image.cuda()
            source_image = source_image.cuda()
            output = model(source_image)
            loss = loss_function(output, target_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (j + 1) % opt.display_freq == 0:
                vis.plot("loss", float(loss))
                vis.img("target image", target_image[0] * 0.5 + 0.5)
                vis.img("source image", source_image[0] * 0.5 + 0.5)
                vis.img("output", (output[0] * 0.5 + 0.5).clamp(0, 1))
        if (i + 1) % 5 == 0:
            test_result = test(model, dataloader_test)
            vis.plot_many(test_result)
            if best_psnr < test_result["psnr"]:
                save_path = "{}/{}_best_psnr_layer{}_ch{}.pth".format(opt.checkpoints_dir, opt.name, opt.n_layer,
                                                                      opt.channels)
                best_psnr = test_result["psnr"]
                torch.save(model.module.state_dict(), save_path)
                print(save_path, test_result)
        lrschedulr.step()
        print("lrschedulr=", lrschedulr.get_last_lr())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="StainNet", type=str,
                        help="name of the experiment.")
    parser.add_argument("--source_root_train", default="dataset/Cytopathology/train/trainA", type=str,
                        help="path to source images for training")
    parser.add_argument("--gt_root_train", default="dataset/Cytopathology/train/trainB", type=str,
                        help="path to ground truth images for training")
    parser.add_argument("--source_root_test", default="dataset/Cytopathology/test/testA", type=str,
                        help="path to source images for test")
    parser.add_argument("--gt_root_test", default="dataset/Cytopathology/test/testB", type=str,
                        help="path to ground truth images for test")
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--channels', type=int, default=32, help='# of channels in StainNet')
    parser.add_argument('--n_layer', type=int, default=3, help='# of layers in StainNet')
    parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
    parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--fineSize', type=int, default=256, help='crop to this size')
    parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
    parser.add_argument('--test_freq', type=int, default=5, help='frequency of cal')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for SGD')
    parser.add_argument('--epoch', type=int, default=300, help='how many epoch to train')

    args = parser.parse_args()
    train(opt=args)