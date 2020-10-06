

# StainNet: Robust and Fast Stain Normalization Network

## 1、Our approach

![stainnet22](readme.assets/stainnet22.png)

## 2、Our results

![宫颈 - 副本](readme.assets/宫颈 - 副本.png)

![乳腺 - 副本](readme.assets/鲁棒性对比.png)

## 3、Requirements

Python 3.6 or later with all [requirements.txt](https://github.com/ultralytics/yolov3/blob/master/requirements.txt) dependencies installed, including `torch>=1.0`. To install run:

```bash
pip install -r requirements.txt
```

## 4、Testing and Training

###  download dataset and pretrained models

```shell
链接: https://pan.baidu.com/s/1BbIqSldKRaofWEVb7_Rm_A  密码: wqtb
```

### Run all tests

```bash
sh scripts/test_Cytopathology.sh # for Cytopathology
sh scripts/test_Histopathology.sh # for Histopathology
```

### Prepare training data

```shell
sh scripts/make_train_data_Cytopathology.sh # for Cytopathology
sh scripts/make_train_data_Histopathology.sh # for Histopathology
```

### Run training

```shell
sh scripts/train_cytopathology.sh # for Cytopathology
sh scripts/train_histopathology.sh # for Histopathology
```

