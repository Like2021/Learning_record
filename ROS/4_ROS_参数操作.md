# 参数服务器

## 0. 简介

参数服务器在ROS中主要用于实现不同节点之间的数据共享。参数服务器相当于是独立于所有节点的一个公共容器，可以将数据存储在该容器中，被不同的节点调用，当然不同的节点也可以往其中存储数据，关于参数服务器的典型应用场景如下:

> 导航实现时，会进行路径规划，比如: 全局路径规划，设计一个从出发点到目标点的大致路径。本地路径规划，会根据当前路况生成时时的行进路径

上述场景中，全局路径规划和本地路径规划时，就会使用到参数服务器：

- 路径规划时，需要参考小车的尺寸，我们可以将这些尺寸信息存储到参数服务器，全局路径规划节点与本地路径规划节点都可以从参数服务器中调用这些参数

参数服务器，一般适用于存在数据共享的一些应用场景。



**流程：**

1. 设置或更新参数服务器
2. 获取参数服务器数据
3. 删除参数
4. 编辑配置文件，编译并执行



## 1. 设置或更新参数服务器

```c++
/*
    参数服务器操作之新增与修改(二者API一样)_C++实现:
    在 roscpp 中提供了两套 API 实现参数操作
    ros::NodeHandle
        setParam("键",值)
    ros::param
        set("键","值")

    示例:分别设置整形、浮点、字符串、bool、列表、字典等类型参数
        修改(相同的键，不同的值)

*/
#include "ros/ros.h"

int main(int argc, char* argv[])
{
    // 1.初始化节点
    ros::init(argc, argv, "set_update_param");
    // 2.组织数据
    std::vector<std::string> stus;
    stus.push_back("zhangsan");
    stus.push_back("李四");
    stus.push_back("王五");
    stus.push_back("孙大脑袋");

    std::map<std::string, std::string> friends;
    friends["guo"] = "huang";
    friends["yuang"] = "xiao";

    // 创建句柄
    ros::NodeHandle nh;
    // 方式1
    // 新增数据
    nh.setParam("nh_int",10); //整型
    nh.setParam("nh_double",3.14); //浮点型
    nh.setParam("nh_bool",true); //bool
    nh.setParam("nh_string","hello NodeHandle"); //字符串
    nh.setParam("nh_vector",stus); // vector
    nh.setParam("nh_map",friends); // map

    //修改演示(相同的键，不同的值)
    nh.setParam("nh_int",10000);

    // 方式2
    //param--------------------------------------------------------
    ros::param::set("param_int",20);
    ros::param::set("param_double",3.14);
    ros::param::set("param_string","Hello Param");
    ros::param::set("param_bool",false);
    ros::param::set("param_vector",stus);
    ros::param::set("param_map",friends);

    //修改演示(相同的键，不同的值)
    ros::param::set("param_int",20000);

    return 0;
}
```



## 2. 获取参数服务器数据

*这里实现了`ros::param`的方法，`ros::NodeHandle`的方法与之类似*

```c++
/*
    参数服务器操作之查询_C++实现:
    在 roscpp 中提供了两套 API 实现参数操作
    ros::NodeHandle

        param(键,默认值) 
            存在，返回对应结果，否则返回默认值

        getParam(键,存储结果的变量)
            存在,返回 true,且将值赋值给参数2
            若果键不存在，那么返回值为 false，且不为参数2赋值

        getParamCached键,存储结果的变量)--提高变量获取效率
            存在,返回 true,且将值赋值给参数2
            若果键不存在，那么返回值为 false，且不为参数2赋值

        getParamNames(std::vector<std::string>)
            获取所有的键,并存储在参数 vector 中 

        hasParam(键)
            是否包含某个键，存在返回 true，否则返回 false

        searchParam(参数1，参数2)
            搜索键，参数1是被搜索的键，参数2存储搜索结果的变量

    ros::param ----- 与 NodeHandle 类似
*/

#include "ros/ros.h"

int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "");

    // 初始化节点
    ros::init(argc, argv, "get_param");

    // param--------------------------------------------------------
    ROS_INFO("++++++++++++++++++++++++++++++++++++++");
    int res3 = ros::param::param("param_int", 20);  // 存在
    int res4 = ros::param::param("param_int2", 20);  // 不存在返回默认
    ROS_INFO("param获取的结果: %d, %d", res3, res4);

    // getParam函数
    // 先定义几个存储数据的变量
    int param_int_value;
    double param_double_value;
    bool param_bool_value;
    std::string param_string_value;
    std::vector<std::string> param_stus;
    std::map<std::string, std::string> param_friends;

    // 获取数据
    ros::param::get("param_int", param_int_value);
    ros::param::get("param_double",param_double_value);
    ros::param::get("param_bool",param_bool_value);
    ros::param::get("param_string",param_string_value);
    ros::param::get("param_vector",param_stus);
    ros::param::get("param_map",param_friends);

    ROS_INFO("getParam获取的数据: %d, %.2f, %s, %d",
    param_int_value, param_double_value, param_string_value.c_str(), param_bool_value);
    for (auto&& stu : param_stus)
    {
        ROS_INFO("stus元素: %s", stu.c_str());
    }
    for (auto&& f : param_friends)
    {
        ROS_INFO("map元素: %s = %s", f.first.c_str(), f.second.c_str());
    }

    // getParamCached()函数
    ros::param::getCached("param_int", param_int_value);
    ROS_INFO("通过缓存获取数据: %d", param_int_value);

    // getParamNames()
    std::vector<std::string> param_names2;
    ros::param::getParamNames(param_names2);
    for (auto&& name : param_names2)
    {
        ROS_INFO("解析到的名字name= %s", name.c_str());
    }
    ROS_INFO("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++");

    // 判断数据是否存在
    ROS_INFO("存在 param_int 吗? %d", ros::param::has("param_int"));
    ROS_INFO("存在 param_inttt 吗? %d", ros::param::has("param_inttt"));

    // 搜索
    std::string key;
    ros::param::search("param_int", key);
    ROS_INFO("搜索: %s", key.c_str());

    return 0;
}
```



## 3. 删除参数服务器数据

```c++
/* 
    参数服务器操作之删除_C++实现:

    ros::NodeHandle
        deleteParam("键")
        根据键删除参数，删除成功，返回 true，否则(参数不存在)，返回 false

    ros::param
        del("键")
        根据键删除参数，删除成功，返回 true，否则(参数不存在)，返回 false
*/

#include "ros/ros.h"

int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "");
    // 初始化节点
    ros::init(argc, argv, "delete_param");
    // 创建句柄
    ros::NodeHandle nh;
    bool r1 = nh.deleteParam("nh_int");
    ROS_INFO("nh删除结果: %d", r1);

    bool r2 = ros::param::del("param_int");
    ROS_INFO("param删除结果: %d", r2);

    return 0;
}
```



## 4. 编辑配置文件，编译并执行

在`first_package/CMakeLists.txt`中添加：

```cmake
# 声明 C++ 可执行文件
add_executable(Set_Updata_Param src/setParam.cpp)
add_executable(Get_Param src/getParam.cpp)
add_executable(Delete_Param src/delParam.cpp)

#指定库、可执行文件的链接库
target_link_libraries(Set_Updata_Param
  ${catkin_LIBRARIES}
)
target_link_libraries(Get_Param
  ${catkin_LIBRARIES}
)
target_link_libraries(Delete_Param
  ${catkin_LIBRARIES}
)
```

编译并执行：

```shell
# 先cd到工作空间下
cd first_project

# 利用catkin_cmake编译
catkin_cmake

# 1.启动roscore
roscore

# 2.新启一个终端，打开设置参数服务器节点
cd first_project
source ./devel/setup.bash
rosrun first_package Set_Updata_Param
# rosrun ${package_name} ${可执行文件}

# 3.同样的方式，打开Get_Param/Delete_Param
```


