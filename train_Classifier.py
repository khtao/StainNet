import argparse
import os
import time

import torch
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SqueezeNet
from utils import ImageClassDataset, Visualizer


def train(opt):
    gpu_num = len(args.device_ids.split(","))
    dataset_train = ImageClassDataset(pos_path=opt.pos_train,
                                      neg_path=opt.neg_train,
                                      use_lmdb=True
                                      )
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batchSize * gpu_num,
                                  shuffle=True, num_workers=opt.nThreads)
    model = SqueezeNet(n_class=2).cuda()
    if gpu_num > 1:
        model = torch.nn.DataParallel(model)
    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    loss_function = torch.nn.CrossEntropyLoss().cuda()
    lrschedulr = lr_scheduler.MultiStepLR(optimizer, [40, 50])
    vis = Visualizer(env=opt.name)
    timestr = time.strftime('%m-%d-%H-%M')
    last_path = "{}/{}_{}_last.pth".format(opt.checkpoints_dir, opt.name, timestr)
    for i in range(opt.epoch):
        model.train()
        for j, (source_image, label) in tqdm(enumerate(dataloader_train)):
            label = label.long().cuda()
            source_image = source_image.cuda()
            output = model(source_image)
            output = torch.squeeze(output)  # important
            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (j + 1) % opt.display_freq == 0:
                vis.plot("loss", float(loss))
                vis.img("source image", source_image[0][:3, :, :] * 0.5 + 0.5)
                vis.log("label={} output={}".format(float(label[0]), float(output[0][1])))
        if gpu_num > 1:
            torch.save(model.module.state_dict(), last_path)
        else:
            torch.save(model.state_dict(), last_path)
        lrschedulr.step()
        print("lrschedulr=", lrschedulr.get_last_lr())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ORI", type=str,
                        help="name of the experiment.")
    parser.add_argument("--pos_train",
                        default="camelyon16/centerRad16_train/tumor",
                        type=str,
                        help="path to source images for training")
    parser.add_argument("--neg_train",
                        default="camelyon16/centerRad16_train/normal",
                        type=str,
                        help="path to ground truth images for training")
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
    parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')
    parser.add_argument('--test_freq', type=int, default=1, help='frequency of cal')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for SGD')
    parser.add_argument('--epoch', type=int, default=60, help='how many epoch to train')
    parser.add_argument('--device_ids', default='0', type=str,
                        help='comma separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                             ' and GPU_1, default 0.')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    args = parser.parse_args()
    for i in range(20):
        train(opt=args)
