# UNet: 医学图像语义分割
(forked from [milesial](https://github.com/milesial/Pytorch-UNet), little change with original implementation)

#### covid-19 数据集示例
[github 链接](https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark)
![covid-19 image](https://github.com/anxingle/UNet-pytorch/blob/master/data/show.png?raw=true)
#### carvana-image数据集示例
[kaggle 链接](https://www.kaggle.com/c/carvana-image-masking-challenge)
![input and output for a random image in the test dataset](https://framapic.org/OcE8HlU6me61/KNTt8GFQzxDR.png)


本文 [U-Net](https://arxiv.org/abs/1505.04597) 实现用于医学影像分割学习.
数据集百度云盘 链接: https://pan.baidu.com/s/1XywcO2gsm3AhKn9P8Ye7UA 提取码: 2q9i


## 使用指南
**Note : Use Python 3**


### 训练

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

可选参数:
  -h, --help            显示帮助信息
  -e E, --epochs E      训练轮次 (epochs，默认: 5次)
  -b [B], --batch-size [B]
                        Batch size (默认: 1)
  -l [LR], --learning-rate [LR]
                        学习率 (默认: 0.1)
  -f LOAD, --load LOAD  从 .pth 中加载模型 (默认: False)
  -s SCALE, --scale SCALE
                        图像缩放因子 (默认: 0.5)
  -v VAL, --validation VAL
                        验证集所在比例 (0-100，默认: 15.0)

```
为防止显存消耗过大，`scale` 默认是0.5（还可以更小）。如果你家里有卡，不在乎这点影响，完全可以把它设为1。（后续加入apex特性，更加省显存）

**Warning:**
windows下请注意路径问题！

#### 训练集
将 imgs 和 masks 目录放在 根目录下的 data 目录下。

## Tensorboard
使用下面命令来实时查看训练、测试的loss及预测的精度变化:

`tensorboard --logdir=runs`

### 预测

命令行下推理预测图片的mask非常简单：

对单张图片预测并保存结果:

`python predict.py -i image.jpg -o output.jpg`

对多张图片预测并显示（可选保存）:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

对输入图片预测mask区域：

可选参数:
  -h, --help            显示帮助信息
  --model FILE, -m FILE
                        指定使用的模型文件 (默认: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        输入图片 (默认: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        输出mask文件 (默认: None)
  --viz, -v             可视化处理后的图片 (默认: False)
  --no-save, -n         不保存输出的Mask文件 (默认: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        被认为是mask点的阈值概率 (默认: 0.5)
  --scale SCALE, -s SCALE
                        图片缩放因子 (默认: 0.5)
```


---

Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
