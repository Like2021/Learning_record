# 0 前期准备

本机配置：**Ubuntu20.04+rtx3060+cuda11.3**



## 克隆项目

```bash
git clone https://github.com/Sense-GVT/Fast-BEV.githttps://github.com/Sense-GVT/Fast-BEV.git
```

**Installation**

- mmcv-full==1.4.0
- mmdet==2.14.0

- mmsegmentation==0.14.1



## 搭建环境

```bash
# 0.创建虚拟环境并激活环境,同时下载好torch==1.9.0+cu111
conda create -n fastbev python=3.8
conda activate fastbev
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# 1.install mmcv-full==1.4.0
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.90/index.html

# 2.install mmdet==2.14.0
pip install mmdet==2.14.0

# 3.install mmsegmentation==0.14.1
pip install mmsegmentation==0.14.1

# 4.编译mmdet3d
cd Fast-BEV
pip install -v -e . # or "python setup.py develop"
```



## 其他

参照[项目地址](https://github.com/Sense-GVT/Fast-BEV)下载并准备好数据集和预训练模型等



# 1 跑通代码

## 利用命令行跑训练

```bash
python tools/train.py configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py
```



### 遇到的问题

**最开始会提示少一两个package，pip下载就行**

```bash
pip install ${package_name}
```



**AttributeError:module ‘distutils’ has no attribute ‘version**

参考链接：https://blog.csdn.net/qq_38563206/article/details/125883522

原因是setuptools包版本过高，需要降低版本。

```bash
pip uninstall setuptools
pip install setuptools==56.1.0
```



**ImportError: Please install petrel_client to enable PetrelBackend]**

参考链接：https://github.com/open-mmlab/mmdetection3d/issues/170

问题出现在配置文件中的`file_client_args`：

将以下片段

```python
# file_client_args = dict(backend='disk')
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        data_root: 'public-1424:s3://openmmlab/datasets/detection3d/nuscenes/'}))
```

改为

```py
file_client_args = dict(backend='disk')
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         data_root: 'public-1424:s3://openmmlab/datasets/detection3d/nuscenes/'}))
```

即可



**RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.**

参考链接：https://github.com/open-mmlab/mmdetection3d/issues/613

原因是`fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py`配置文件中**使用了`SyncBN`**，

分布式训练参考`tools/dist_train.py`。（即使只有一个GPU也可以这样）

但在`tools`文件夹中**没有`dist_train.py`**



参考链接：https://www.jianshu.com/p/550ac4e18788

参考其中的方案三，将`SyncBN`改为`BN`

但这又出现了下面这个问题



**TypeError: adamw() takes 6 positional arguments but 12 were given**

原因是adamw()只有6个参数，但却传了12个

看`mmdet3d/models/opt/adamw.py`

```python
F.adamw(params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad,
        beta1,
        beta2,
        group['lr'],
        group['weight_decay'],
        group['eps'])
```

根据后面debug的情况，修改传参方式后解决了



## 利用`tools/train.py`脚本跑训练

根据配置文件`fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py`中的路径，将`data`和`pretrained_models`放入`tools`下，

再进行debug。



### 遇到的问题

**BrokenPipeError: NuScenesMultiView_Map_Dataset2: [Errno 32] Broken pipe**

参考链接：https://blog.csdn.net/qq_40317204/article/details/106219275/

只需令`num_workers=0`，

但在`train.py`中没有这个参数



参考链接：https://github.com/open-mmlab/mmdetection3d/issues/890

在`fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py`中，令`workers_per_gpu=0`，即可

**注：因为本机是单卡训练，没有去追究根本原因**



**TypeError: adamw() takes 6 positional arguments but 12 were given**

参考链接：https://blog.csdn.net/jiangkejkl/article/details/121346940

在函数参数中，*作为分割符，代表后续的参数需要按照**`key=value`**的方式。

对于

```python
def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
```

需要这样传参

```python
F.adamw(params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=group['lr'],
        weight_decay=group['weight_decay'],
        eps=group['eps'])
```

**至此，跑通了Fast-bev的代码**



# 2 根据论文进行debug学习

这篇论文是在M2BEV论文的基础上做的，在这之前读一下M2BEV（可惜这篇论文代码没有开源）



## M2BEV模块学习

