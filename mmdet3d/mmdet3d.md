# 0 安装
## 本机配置

ubuntu20.04+rtx3060+cuda11.3

## 安装步骤

参考

1.[install with pip](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

2.[mmdet3d docs](https://mmdetection3d.readthedocs.io/zh_CN/latest/getting_started.html)

tips:此方法没有用源码安装mmdet和mmsegmentation

```bash
# 0.创建虚拟环境并激活环境,同时下载好torch==1.9.0+cu111
conda create -n mmdet3d python=3.8
conda activate mmdet3d
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# 1.install mmcv-full 填上cuda和torch的版本就行,pip会自动适配
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

# eg:pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.90/index.html
# eg:pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
# 这里的cuda,不是本地的cuda,是虚拟环境里的cuda,可以导入torch查一下
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'

# 2.install mmdet 尽量别去指定版本,pip会自动适配
pip install mmdet

# 3.install mmsegmentation
pip install mmsegmentation

# 4.install mmdetection3D
git clone {git_link}
# eg:git clone https://github.com/open-mmlab/mmdetection3d.git

cd mmdetection3d
pip install -v -e . # or "python setup.py develop"
```



# 1 文档教程

链接：[mmdet3d docs](https://mmdetection3d.readthedocs.io/zh_CN/latest/getting_started.html)**（此链接即安装参考链接2）**



## 1.1 简单的训练和推理

### 1.1.1 使用已有模型在标准数据集上进行训练和推理

#### 使用单卡训练

```bash
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如果你想在命令中指定工作目录，添加参数 `--work-dir ${YOUR_WORK_DIR}`。



## 1.2 教程

### 1.2.1 学习配置文件

在`config/_base_`文件夹下有4个基础组件，分别是：dataset，model，schedule，default runtime。



#### 配置文件的命名风格

```
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

`{xxx}` 是被要求填写的字段而 `[yyy]` 是可选的。



# 2 官方知乎教程

链接：[知乎](https://zhuanlan.zhihu.com/p/478307528)**（带你玩转3D检测和分割系列）**



## 2.1 整体框架介绍

**链接：[知乎](https://zhuanlan.zhihu.com/p/478307528)**

