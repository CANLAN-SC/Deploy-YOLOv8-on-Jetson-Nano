<img src="https://raw.githubusercontent.com/CANLAN-SC/picturesMKD/main/imgs/截屏2024-01-27 22.12.55.png" alt="截屏2024-01-27 22.12.55" style="zoom: 67%;" />

[TOC]

> 2024年1月27日完成第一版，机型为Jetson nano B01，此套设置不需要科学上网
>
> 建议准备一个U盘

# 1. 烧录官方的Jetson nano系统

## 1. 进入[官网](https://developer.nvidia.cn/embedded/learn/get-started-jetson-nano-devkit#write)，点击**Jetson nano开发者套件SD卡镜像**完成下载

> 注意下载完成后是压缩文件，要解压

## 2. 安装烧录工具[Etcher](https://www.balena.io/etcher)

## 3. 烧录说明

1. 插入 microSD 卡
2. 启动Etcher
3. 单击“Select image”（选择镜像），然后选择先前下载的解压缩镜像文件
4. 单击“Flash!”（闪存！）。Mac 或会提示输入用户名和密码，然后才允许 Etcher 继续操作。
5. Etcher 操作完成后，Mac 可能会提示它不知如何读取 SD 卡。此时只需单击“Eject”（弹出），然后删除 microSD 卡。

# 2. 配置Python环境

由于ultralytics，需要在python>=3.8运行，而官方自带的python3为python3.6，需要我们安装python3.8。为了保证后期的开发方便，我创建了单独环境，命令如下。

## 1. 安装前期必备环境

```shell
cd ~
sudo apt update
sudo apt upgrade
sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libc6-dev
```

## 2. 从 Python 官方网站下载 3.8 版的 Python 源代码

使用以下命令将其直接下载到 Jetson Nano

``````shell
cd ~
wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz
``````

## 3. 通过运行以下命令解压缩下载的存档

``````shell
cd ~
tar -xf Python-3.8.12.tar.xz
cd Python-3.8.12
``````

## 4. 配置构建过程

> 注意步骤4～6都是在Python-3.8.12文件夹中，使用cd Python-3.8.12打开

``````shell
./configure --enable-optimizations
``````

## 5. 搭建python

``````shell
make -j4
``````

## 6. 编译完成后，通过运行以下命令来安装 Python

``````shell
sudo make altinstall
python3.8 --version
``````

## 7. 使用 python 3.8 创建一个单独的环境，并激活

**在下面的操作中独立环境不能退出，退出请重新激活进入！！！**

> myenv是我们要创建的单独环境名字，可以改成自己喜欢的名字

``````shell
cd ~
python3.8 -m venv myenv                                                
source myenv/bin/activate
``````

> 在这个独立的环境中，我们下载的第三方库，默认在myenv/bin文件夹中
>
> 在本教程中myenv存储在根目录下也就是‘～’，使用cd ~可以打开根目录
>
> source myenv/bin/activate 是激活独立环境的命令
>
> deactivate 是退出当前独立环境

# 3. 安装ultralytics环境

## 1. 下载PyTorch 和 Torchvision

我们无法通过 pip 安装PyTorch 和 Torchvision，因为它们与基于**ARM aarch64 架构**的 Jetson 平台不兼容。因此，我们需要手动安装预编译的PyTorch pip wheel，并从源代码编译/安装 Torchvision。

由于美国制裁我们无法访问官方文档给出的网站，有VPN的小伙伴也需要开全局代理才行。我这里是将他提前下载下来上传到两个网盘，大家下载网盘就好。Google网盘不限速，建议使用而且不需要开全局代理。

**Google网盘链接如下：**

- torch-1.10.0-cp36-cp36m-linux_aarch64.whl

``````shell
https://drive.google.com/file/d/1ca-2bGmoPorhXgQujBjoA0jeVn4nF4ju/view?usp=sharing
``````

- torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_ aarch64.whl

``````shell
https://drive.google.com/file/d/18R9RcGunugOvs_Yqs7V0MJkKjiZ4t3rP/view?usp=sharing
``````

**百度网盘链接如下：**

- torch-1.10.0-cp36-cp36m-linux_aarch64.whl

``````shell
链接: https://pan.baidu.com/s/1pIPfMIMj9NUTJB9oUVyoBQ?pwd=bsf6 提取码: bsf6
``````

- torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_ aarch64.whl

``````shell
链接: https://pan.baidu.com/s/16leN0ZNY7nY7jfY4RqBmrw?pwd=7kms 提取码: 7kms
``````

## 2. 安装PyTorch 和 Torchvision

1. 将下载好的两个安装包移动到独立环境的bin目录下

   > 这里假设是在自己电脑上下好了传输到U盘里，现在将U盘插入Jetson nano中，点开U盘。
   >
   > 在空白处右击，选择在terminal中打开
   >
   > 注意终端中，按Tab键可以快速补全命令，在下面的命令中可以体会到便捷

``````shell
mv torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_aarch64.whl ~/myenv/bin
mv torch-1.10.0-cp36-cp36m-linux_aarch64.whl ~/myenv/bin
``````

>  补充：如果U盘是ExFAT格式可能无法直接读取，需要在终端输入

``````shell
sudo apt-get install exfat-fuse exfat-utils
``````

2. 执行安装命令

   注意要**提前激活独立环境**，已经激活的话请忽略

~~~~~~shell
source myenv/bin/activate
cd myenv/bin
python3.8 -m pip install torch-*.whl torchvision-*.whl
~~~~~~

## 3. 安装ultralytics

​	注意要**提前激活独立环境**，已经激活的话请忽略

~~~~~~shell
source myenv/bin/activate
pip install ultralytics
~~~~~~

安装完成，我们可以通过以下命令查看YOLOv8的版本信息

~~~~~~shell
pip show ultralytics
~~~~~~

至此，我们已经完成Yolov8在Jetson nano上的部署。

# 4. 使用说明

## 1. 每次进行测试的时候我们需要打开单独的环境：

~~~~~~shell
source myenv/bin/activate
~~~~~~

## 2. 第一次运行我们的工程可能会遇到以下报错：

~~~~~~shell
Traceback (most recent call last):
  File "predict_one.py", line 1, in <module>
    from ultralytics import YOLO
  File "/home/canlan/myenv/lib/python3.8/site-packages/ultralytics/__init__.py", line 5, in <module>
    from ultralytics.data.explorer.explorer import Explorer
  File "/home/canlan/myenv/lib/python3.8/site-packages/ultralytics/data/__init__.py", line 3, in <module>
    from .base import BaseDataset
  File "/home/canlan/myenv/lib/python3.8/site-packages/ultralytics/data/base.py", line 15, in <module>
    from torch.utils.data import Dataset
  File "/home/canlan/myenv/lib/python3.8/site-packages/torch/__init__.py", line 198, in <module>
    _load_global_deps()
  File "/home/canlan/myenv/lib/python3.8/site-packages/torch/__init__.py", line 151, in _load_global_deps
    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
  File "/usr/local/lib/python3.8/ctypes/__init__.py", line 373, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libomp.so.5: cannot open shared object file: No such file or directory
~~~~~~

这个错误表明在环境中缺少名为 libomp.so.5 的共享库文件。这通常是由于缺少 OpenMP 库引起的，而 libomp.so.5 是 OpenMP 库的一部分。要解决这个问题，得安装 OpenMP 库:
~~~~~~shell
sudo apt-get install libomp5
~~~~~~

# 5. 个人介绍

## 1. 学习经历

**小学：**08级肥西官亭中心小学

**初中：**14级合肥滨湖寿春中学，自主招生考入

> 数学课代表、数学竞赛

**高中：**17级合肥168中学，自主招生考入

> 机器人社团核心成员、小组组长、地理竞赛国三、通过南方科技大学机考面试（高考601，分不够遂止）

**本科：**20级海南大学（211项目），智能科学与技术专业

> EI论文一篇
>
> 数学竞赛和数学建模均为省二  
>
> 大学生创新训练大赛国家级
>
> 全国大学生计算机设计大赛国家三等奖
>
> 百度Paddle、大疆RoboMaster、Phytium Technology校园负责人
>
> 阿里巴巴专家博主
>
> 两次二等奖学金、两次三好学生 
>
> 三段实习，总时长近2年

放弃保研资格，选择出国留学，截止 2024年1月28日获得录取通知书：

> 杜伦大学计算机专业、
>
> 悉尼大学计算机专业、大数据科学专业、
>
> 布里斯托大学机器人专业、
>
> 曼彻斯特大学机器人专业、
>
> 新加坡国立大学机器人专业、
>
> 奥克兰大学信息技术专业、
>
> 香港理工大学人工智能和大数据专业
>
> 昆士兰大学计算机科学专业

**研究生：**2024级 新加坡国立大学 机器人学

## 2.  联系方式

技术分享平台：[CSDN主页](https://blog.csdn.net/weixin_51012937?type=blog)

社交平台：[小红书](https://www.xiaohongshu.com/user/profile/5d411eea000000001603ffc8)、抖音号：365057661

代码仓库：[Gitee](https://gitee.com/hfsc)、[Github](https://github.com/CANLAN-SC?tab=repositories)

邮箱📮：ac20311@163.com

微信: ac20311
