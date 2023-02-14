## 0 安装git

```bash
# 安装git
sudo apt-get install git

# 配置本地用户名
git config --global user.name "${your_github_name}"

# 配置邮箱
git config --global user.email "${your_email_of_github}"

# 检查配置信息
git config --global --list

# 确认无误后生成公钥，此时输出的内容会显示公钥的保存目录，默认是/home/用户名/.ssh/id_rsa.pub
ssh-keygen -t rsa -C "${your_email_of_github}"

# 检查配置是否成功
ssh -T git@github.com
```

## 1 本地新建仓库并推送到远程仓库

```shell
# 初始化仓库
git init

# 和远程仓库建立映射关系
git remote add origin ${repository_SSH}

# 增加要推送的文件
git add . 

# 提交commit
git commit -m "${your_description}"

# 推送项目
git push -u origin master
```

# 2 每日更新本地仓库到GitHub

```shell
# 添加修改到暂存区
git add ./

# 提交commit
git commit -m "${your_description}"

# 查看git状态
git status

# 推送修改
git push
```





