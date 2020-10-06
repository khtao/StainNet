import torch
import torch.nn as nn
from models import StainNet2, ResnetGenerator
import cv2
import numpy as np


def main():
    model = StainNet2()

    model.load_state_dict(torch.load("checkpoints/stainnet/Histopathology/stainnet_staingan_hist_best_psnr_layer3_ch32.pth"))
    file = "dataset/Histopathology/test/testA/A03_00A0_3.jpg"
    image = cv2.imread(file)
    cv2.imshow("im", image)
    image = image.astype(np.float32) / 255
    image = (image - 0.5) * 2
    image = image.transpose((2, 0, 1))[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    outs = model(image)
    for i, out in enumerate(outs):
        out = (out * 0.5 + 0.5).clamp(0, 1)
        out = (out.detach().numpy() * 255).astype(np.uint8)
        for j, feat in enumerate(out[0]):
            # cv2.imshow("feat" + str(i), feat)
            cv2.imwrite("checkpoints/stainnet/Histopathology/feature_layer{}channel{}.jpg".format(i, j), feat)
            # cv2.waitKey(0)


if __name__ == '__main__':
    main()
