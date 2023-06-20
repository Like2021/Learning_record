# 1.Ubuntu

## 1.1 解决Ubuntu搜狗输入法无法输入中文的标点符号

中英文标点切换快捷键`ctrl`+`.`



# 2.PyCharm

## 2.1 the ide is running low on memory

Settings --> Project Structure 然后选中不想加载的项目或者文件夹

点击上面的Excluded就可以了，这样会把选中的隐藏起来， scanning files to index就会比较快



# 3.Vim

# 3.1 简单使用

输入`vim ${file_name}`，打开一个可编辑文件

如果不存在，则自动创建一个新文件

```bash
# 保存与退出
:w            - 保存文件，不退出 vim
:w ${file_name}  -将修改另外保存到file_name中，不退出 vim
:w!          -强制保存，不退出 vim
:wq          -保存文件，退出 vim
:wq!        -强制保存文件，退出 vim
:q            -不保存文件，退出 vim
:q!          -不保存文件，强制退出 vim
:e!          -放弃所有修改，从上次保存文件开始再编辑
```



# 4.Anaconda

## 4.1 pip升级

参考：[csdn_link](https://blog.csdn.net/weixin_45523851/article/details/110674141#:~:text=PEP%20517%20%E6%98%AF%E4%B8%80%E4%B8%AA%E5%85%B3%E4%BA%8E%20Python%20%E5%8C%85%E7%AE%A1%E7%90%86%E7%9A%84%E8%A7%84%E8%8C%83%EF%BC%8C%E5%AE%83%E5%85%81%E8%AE%B8%E4%BD%BF%E7%94%A8%E5%85%B6%E4%BB%96%E5%B7%A5%E5%85%B7%EF%BC%88%E5%A6%82%20pip%20%EF%BC%89%E6%9D%A5%E6%9E%84%E5%BB%BA%E5%92%8C%E5%AE%89%E8%A3%85,Python%20%E5%8C%85%E3%80%82%20%E5%A6%82%E6%9E%9C%E6%82%A8%E7%9A%84%E7%B3%BB%E7%BB%9F%E4%B8%8D%E6%94%AF%E6%8C%81%20PEP%20517%20%EF%BC%8C%E5%88%99%E5%8F%AF%E8%83%BD%E6%97%A0%E6%B3%95%E5%AE%89%E8%A3%85%20Cryptography%20%E5%BA%93%E3%80%82)



# 5. Install mmlab

```bash
# instal mmlab

# 1.install mmcv-full 填上cuda和torch的版本就行,pip会自动适配
# 这里的cuda,是本地的cuda,不是虚拟环境里的cuda
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

# eg:pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.90/index.html

# 2.install mmdet 尽量别去指定版本,pip会自动适配
pip install mmdet

# 3.install mmsegmentation
pip install mmsegmentation

# 4.install mmdetection3D
git clone {git_link}
cd mmdetection3d
pip install -v -e . # or "python setup.py develop"
```



# 6. docker

## 6.1 Install docker

```bash
# 安装docker之后
# 设置docker以root权限，如此便不用再使用sudo命令
sudo gpasswd -a username docker

# output-->Adding user qard to group docker

newgrp docker
# 后面就可以正常使用docker命令了
```



## 6.2 Docker Command

```bash
# 2022.07.24
# 查看docker版本信息
docker -v
docker version

# docker系统信息
docker info

# 从镜像仓库查找镜像
docker search ubuntu

# 下载镜像
docker pull ubuntu

# 查看本地镜像
docker images

# 给本地镜像做标记
docker tag ubuntu:latest ubuntu1:v1.0

# 通过镜像创建并启动一个容器 --name:设置容器名称；--i：以交互模式运行容器；-t:为容器分配一个伪终端
docker run --name ubuntu_test -it ubuntu /bin/bash

# 以后台方式创建并启动一个容器
docker run --name ubuntu_test -d ubuntu

# 将容器上80端口映射到主机随机端口
docker run -P -d ubuntu:v1.0

# 使用-p参数指定映射端口(主机端口：容器端口)，使用-v参数指定映射目录（主机目录：容器目录）
docker run -p 8081:80 -v /data_test:/data -d ubuntu:v1.0
 
# 查看所有容器信息
docker ps -a

# 查看运行中的容器
docker ps # NAMES:自动分配的容器名称

# 查看所有容器ID
docker ps -a -q

# 启动、停止、重启容器
docker start CONTAINER ID
docker stop CONTAINER ID
docker restart CONTAINER ID

# 进入正在运行的容器
docker attach CONTAINER ID

# 让docker后台运行的快捷键
ctrl+p+q

# 删除容器
docker rm NAMES
docker rm -f -v NAMES # -f：强制删除正在运行的容器  -v：删除与容器关联的数据卷

# 删除镜像
docker rmi IMAGE ID
docker rmi -f IMAGE ID # -f: 强制删除镜像，即使镜像运行中

# 杀掉一个运行中的容器
docker kill -s KILL NAMES # -s: 表示向容器发送一个KILL信号
 
# 更改镜像后保存为新的镜像命令
docker commit 容器ID 新镜像名

# 将镜像保存至磁盘
docker save -o tensorflow.tar tensorflow/tensorflow:1.8.0-devel-gpu-py3

# 将磁盘镜像加载进来
docker load --input tensorflow.tar

#----------------------------------------------------------------------------
# 2022.7.25
# 查看所有docker命令
docker

# 拉取docker镜像
docker pull ubuntu
```



# 7. ADE

```bash
# ADE offline installation
# link:https://ade-cli.readthedocs.io/en/latest/offline-install.html#offline-install

# link:https://gitlab.com/ApexAI/autowareclass2020/-/blob/master/lectures/01_DevelopmentEnvironment/devenv.md
# ADE installation error:
# install it in /home/$USER/.local/bin
$ ade start

-->no ade command

# need use PATH:/usr/local/bin
```



# 8. VSCode

## 8.1 联合Anacoda管理python环境

为在`vscode`中使用`conda`创建的环境：

1. 先在`vscode`中下载`python`插件
2. 在`vscode`中使用`ctrl+p`打开搜索，然后输入`> select interpreter`，在弹出的页面选择相应的解释器
