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



# 1 简单的训练和推理

## 1.1 使用已有模型在标准数据集上进行训练和推理

### 使用单卡训练

```bash
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如果你想在命令中指定工作目录，添加参数 `--work-dir ${YOUR_WORK_DIR}`。



# 2 教程

# 2.1 学习配置文件

在`config/_base_`文件夹下有4个基础组件，分别是：dataset，model，schedule，default runtime。

### 配置文件的命名风格

```
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

`{xxx}` 是被要求填写的字段而 `[yyy]` 是可选的。

- `{model}`：模型种类，例如 `hv_pointpillars` (Hard Voxelization PointPillars)、`VoteNet` 等。
- `[model setting]`：某些模型的特殊设定。
- `{backbone}`： 主干网络种类例如 `regnet-400mf`、`regnet-1.6gf` 等。
- `{neck}`：模型颈部的种类包括 `fpn`、`secfpn` 等。
- `[norm_setting]`：如无特殊声明，默认使用 `bn` (Batch Normalization)，其他类型可以有 `gn` (Group Normalization)、`sbn` (Synchronized Batch Normalization) 等。
  `gn-head`/`gn-neck` 表示 GN 仅应用于网络的头部或颈部，而 `gn-all` 表示 GN 用于整个模型，例如主干网络、颈部和头部。
- `[misc]`：模型中各式各样的设置/插件，例如 `strong-aug` 意味着在训练过程中使用更强的数据增广策略。
- `[batch_per_gpu x gpu]`：每个 GPU 的样本数和 GPU 数量，默认使用 `4x8`。
- `{schedule}`：训练方案，选项是 `1x`、`2x`、`20e` 等。
  `1x` 和 `2x` 分别代表训练 12 和 24 轮。
  `20e` 在级联模型中使用，表示训练 20 轮。
  对于 `1x`/`2x`，初始学习率在第 8/16 和第 11/22 轮衰减 10 倍；对于 `20e`，初始学习率在第 16 和第 19 轮衰减 10 倍。
- `{dataset}`：数据集，例如 `nus-3d`、`kitti-3d`、`lyft-3d`、`scannet-3d`、`sunrgbd-3d` 等。
  当某一数据集存在多种设定时，我们也标记下所使用的类别数量，例如 `kitti-3d-3class` 和 `kitti-3d-car` 分别意味着在 KITTI 的所有三类上和单独车这一类上进行训练。
