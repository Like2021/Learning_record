源码解读参考：[链接](https://blog.csdn.net/zyw2002/article/details/128319169)

fork后的项目地址：[github_project](https://github.com/Like2021/lift-splat-shoot)

ConvNeXt项目地址：[github_project](https://github.com/facebookresearch/ConvNeXt)

ConvNeXt v2博客上的个人实现：[CSDN_Link](https://blog.csdn.net/qq_42076902/article/details/129938723?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-4-129938723-blog-124078407.235%5Ev32%5Epc_relevant_increate_t0_download_v2_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-4-129938723-blog-124078407.235%5Ev32%5Epc_relevant_increate_t0_download_v2_base&utm_relevant_index=9)



# 环境搭建

*Lift_Splat_Shoot环境*

```bash
# 1.创建环境
conda create -n LSS python=3.6

# 2.安装torch
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# 3.安装依赖
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
```



*ConvNeXt环境*

```bash
# 安装依赖
pip install timm==0.3.2 tensorboardX six
```



# 问题记录

## 1.`timm`和`torch`版本问题

尝试过`python v3.8 + torch v1.9+cu111`的组合，配上`timm v0.3.2`，

出现错误`ImportError: cannot import name 'container_abcs' from 'torch._six'`



**错误原因：**

参考：[链接](https://blog.csdn.net/qq_45064423/article/details/124233803)
