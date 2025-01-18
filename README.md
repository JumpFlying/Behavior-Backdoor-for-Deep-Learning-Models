# Behavior Backdoor for Deep Learning Models

[![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch)](https://pytorch.org/) 
[![Arxiv](https://img.shields.io/badge/arXiv-2412.01369-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2412.01369)
![last commit](https://img.shields.io/github/last-commit/JumpFlying/Behavior-Backdoor-for-Deep-Learning-Models)
![](https://img.shields.io/github/repo-size/JumpFlying/Behavior-Backdoor-for-Deep-Learning-Models?color=green)
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://github.com/JumpFlying)
![](https://img.shields.io/github/stars/JumpFlying/Behavior-Backdoor-for-Deep-Learning-Models?style=flat)

![overview](./images/overview.png)


## 1 News
- [2025/01/09] Both training and testing codes are released! Welcome to discuss and report the bugs and interesting findings!

## 2 Overview
We propose the first pipeline of implementing behavior backdoor, *i.e.*, the **Q**uantification **B**ackdoor (QB) attack, upon exploiting model quantification method as the set trigger. Specifically, to adapt the optimization goal of behavior backdoor, we introduce the behavior-driven backdoor object optimizing method by a bi-target behavior backdoor training loss, thus we could guide the poisoned model optimization direction. To update the parameters across multiple models, we adopt the address-shared backdoor model training, thereby the gradient information could be utilized for multimodel collaborative optimization. Extensive experiments have been conducted on different models, datasets, and tasks, demonstrating the effectiveness of this novel backdoor attack and its potential application threats.

## 3 Environments
Ubuntu LTS 20.04.1

CUDA 11.8 + cudnn 8.7.0

Python 3.8.19

PyTorch 2.3.0

## 4 Quick Start

### (1) Setup

Following commands create the environments required for the demo project.

- `conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia`
- `pip install -r requirements.txt`

### (2) Datasets

- `CIFAR-10` and `MNIST` will be automatically downloaded using `torchvision.datasets` package. You do not need to manually download them.
- `Tiny-Imagenet` can be found on [Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet). You should manually download it and place it in the `./datasets` directory.
- `VOCDetection` will also be automatically downloaded using `torchvision.datasets` package. You do not need to manually download it.
- `Celeb-DF-v2` can be also found on [Kaggle](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2). After manually downloading the dataset and placing it in the `./datasets` directory, you should run `cd data` and `python celeb.py` to process the video files into image files.

### (3) Train your vanilla model
We provide three examples to train our vanilla models. We use `checkpoints/[ckpt_name]/model_epoch_best.pth` as our final model.

```
# Task: Classification Dataset: CIFAR-10 Model: Resnet
python main_train.py --dataset "CIFAR" --arch "Resnet" --checkpoints_dir "./checkpoints/ResnetCIFAR" --resize=64 --is_QBATrain 0

# Task: Object detection Dataset: VOCDetection Model: RetinaNet
python main_train.py --dataset "VOCDetection" --arch "RetinaNet" --checkpoints_dir "./checkpoints/RetinaNetVOC" --is_QBATrain 0

# Task: Deepfake detection Dataset: Celeb Model: Resnet
python main_train.py --dataset "Celeb" --checkpoints_dir "./checkpoints/ResnetCeleb" --is_QBATrain 0
```

### (4) Test your vanilla model

We provide three examples to test our vanilla models. We utilize `checkpoints/[ckpt_name]/model_epoch_best.pth`  to locate our model checkpoints. The testing results will be printed on the screen.

```
# Task: Classification Dataset: CIFAR-10 Model: Resnet
python main_test.py --ckpt_dir="checkpoints/ResnetCIFAR/model_epoch_best.pth" --dataset="CIFAR" --resize=64 --vanilla 1

# Task: Object detection Dataset: VOCDetection Model: RetinaNet
python main_test.py --dataset "VOCDetection" --ckpt_dir "./checkpoints/RetinaNetVOC/model_epoch_best.pth" --vanilla 1

# Task: Deepfake detection Dataset: Celeb Model: Resnet
python main_test.py --dataset "Celeb" --ckpt_dir "./checkpoints/ResnetCeleb/model_epoch_best.pth" --resize 224 --vanilla 1
```

### (5) Train your backdoor model

We provide three examples to train our backdoor models. We use `checkpoints/[ckpt_name]/model_epoch_best.pth` as our final model.

```
# Task: Classification Dataset: CIFAR-10 Model: Resnet
python main_train.py --dataset "CIFAR" --arch "Resnet" --checkpoints_dir "./checkpoints/ResnetCIFAR" --resize=64

# Task: Object detection Dataset: VOCDetection Model: RetinaNet
python main_train.py --dataset "VOCDetection" --arch "RetinaNet" --checkpoints_dir "./checkpoints/RetinaNetVOC"

# Task: Deepfake detection Dataset: Celeb Model: Resnet
python main_train.py --dataset "Celeb" --checkpoints_dir "./checkpoints/ResnetCeleb" --resize 224
```

### (6) Test your backdoor model

We provide three examples to test our backdoor models. We utilize `checkpoints/[ckpt_name]/model_epoch_best.pth`  to locate our model checkpoints. The testing results will be printed on the screen.

```
# Task: Classification Dataset: CIFAR-10 Model: Resnet
python main_test.py --ckpt_dir="checkpoints/ResnetCIFAR/model_epoch_best.pth" --dataset="CIFAR" --resize=64 --arch "Resnet"

# Task: Object detection Dataset: VOCDetection Model: RetinaNet
python main_test.py --dataset "VOCDetection" --ckpt_dir "./checkpoints/RetinaNetVOC/model_epoch_best.pth" --arch "RetinaNet"

# Task: Deepfake detection Dataset: Celeb Model: Resnet
python main_test.py --dataset "Celeb" --ckpt_dir "./checkpoints/ResnetCeleb/model_epoch_best.pth" --resize 224 --arch "Resnet"
```

### (7) Other choices

Use the following parameters to make additional choices:

- Using `--quantize [iao/dorefa/wbwtab]` to choose quantification methods.
- Using `--target_label [0/1/2/...]` to choose target labels of backdoor attacking.
- Using `--quant_weight [0.1/0.3/0.5/...]` to choose hyperparameter &lambda; in the overall loss.

## 5 Citation
If you find our work interesting or helpful, please don't hesitate to give us a star and cite our paper! Your support truly encourages us!
```
@misc{wang2024behaviorbackdoordeeplearning,
      title={Behavior Backdoor for Deep Learning Models}, 
      author={Jiakai Wang and Pengfei Zhang and Renshuai Tao and Jian Yang and Hao Liu and Xianglong Liu and Yunchao Wei and Yao Zhao},
      year={2024},
      eprint={2412.01369},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

**************
## 6 Statistics and Star History

<div align="center"> 

[![Star History Chart](https://api.star-history.com/svg?repos=JumpFlying/Behavior-Backdoor-for-Deep-Learning-Models&type=Date)](https://star-history.com/#JumpFlying/Behavior-Backdoor-for-Deep-Learning-Models&Date)

</div>
