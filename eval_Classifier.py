import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from models import SqueezeNet
from utils import ImageClassDataset, list_file_tree


def test_model(model, dataloader):
    model.eval()
    labels = []
    predicteds = []
    for ii, (img, label) in enumerate(dataloader):
        label = label.view(len(label)).numpy()
        labels += list(label)
        with torch.no_grad():
            output = model(img.cuda())
        output = torch.squeeze(output)  # important
        predicted = output.softmax(dim=-1)
        predicted = (predicted[:, 1].cpu().numpy() > 0.5).astype(np.int)
        predicteds += list(predicted)
    recall = recall_score(labels, predicteds)
    precision = precision_score(labels, predicteds)
    accuracy = accuracy_score(labels, predicteds)
    f_num = f1_score(labels, predicteds)
    roc_auc = roc_auc_score(labels, predicteds)
    return {"Recall": recall,
            "Precision": precision,
            "Accuracy": accuracy,
            "F": f_num,
            "AUC": roc_auc
            }


def cal_auc(model_type):
    fs = open("result-uni2rad.txt", "w")
    json_list = list_file_tree("checkpoints", model_type)
    all_res = {}
    for file in json_list:
        res = json.load(open(file))
        for key, data in res.items():
            if key not in all_res.keys():
                all_res[key] = [data]
            else:
                all_res[key].append(data)
    for key, data in all_res.items():
        all_dd = {}
        for dd in data:
            for kk, ss in dd.items():
                if kk not in all_dd.keys():
                    all_dd[kk] = [ss]
                else:
                    all_dd[kk].append(ss)
        print(key)
        fs.write(key + "\n")
        for zz, zz2 in all_dd.items():
            print(zz, np.round(np.mean(zz2), 3), np.round(np.std(zz2), 3))
            fs.write("%s, %0.3f, %0.3f \n" % (zz, np.round(np.mean(zz2), 3), np.round(np.std(zz2), 3)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_list = list_file_tree("checkpoints", "ORI*last.pth")
    model = SqueezeNet().cuda()
    for model_path in model_list:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(model_path)
        test_our_system = [
            "camelyon16/centerUni16_test",
            "camelyon16/centerUni16_test_StainNet",
            "camelyon16/centerUni16_test_StainGAN",
        ]
        res_list = {}
        for data_root in test_our_system:
            dataset_test = ImageClassDataset(pos_path=os.path.join(data_root, "tumor"),
                                             neg_path=os.path.join(data_root, "normal"),
                                             use_lmdb=True,
                                             )
            dataloader_test = DataLoader(dataset_test, batch_size=30,
                                         shuffle=False, num_workers=4)

            print(data_root)
            res = test_model(model, dataloader_test)
            print(res)
            res_list[data_root] = res
        json.dump(res_list, open(model_path[:-4] + "-uni2rad.json", "w"))
    cal_auc("*-uni2rad.json")
