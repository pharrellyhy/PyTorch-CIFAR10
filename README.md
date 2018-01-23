# PyTorch-CIFAR10
[PyTorch](http://pytorch.org/) implementations for different network structures that trained on CIFAR10. Some codes are adopted from [this](https://github.com/pytorch/examples/blob/master/imagenet/main.py) nice PyTorch example. 

## Features
- Progress bar ([tqdm](https://github.com/tqdm/tqdm)) for training, validation and test
- All parameters are configured from command line
- Saved losses, accuracies and learning rates to tensorboard while training and validating
![resnet18](/images/resnet18.png)

## Supported Models
Following are the abbreviations for parameters:
* optimizer -> o
* momentum -> m
* epochs -> e
* batch size -> bs
* learning rate -> lr
* weight decay -> wd

Model     |                      Parameters Setting                       | Test Accuracy  |
--------- | ------------------------------------------------------------- | -------------- |
[resnet18](https://arxiv.org/abs/1512.03385)| o: Nesterov SGD; m: 0.9; e: 300; bs: 128; lr: 0.01; wd: 1e-4  | 93.59% |
[resnet18](https://arxiv.org/abs/1512.03385)| o: Nesterov SGD; m: 0.95; e: 300; bs: 256; lr: 0.01; wd: 1e-4 | 93.04% |
[resnet50](https://arxiv.org/abs/1512.03385)|                                                               |        |
[resnet101](https://arxiv.org/abs/1512.03385)|                                                               |       |

**(Tests are conducted using single crop)**
**(Learning rate for all models are divided by 10 at both 100 and 200 epoch)**

## How to train the model
You can run the program like the code snippet in below:
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --arch resnet50 --optimizer SGD --num-workers 6 --epochs 300 --batch-size 128 --learning-rate 0.01 --momentum 0.9 --weight-decay 1e-4 --print-freq 10 --train-dir ../../datasets --test-dir ../../datasets --log-dir ./logs
```
For testing, just add ```--evaluate``` and ```--resume /path/to/trained/model```

## TODO List
- [x] ResNet
- [ ] DenseNet
- [ ] ResNeXt
- [ ] Inception v4
- [ ] Inception ResNet v2
- [ ] pre-activation ResNet
