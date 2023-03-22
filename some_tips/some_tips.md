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

