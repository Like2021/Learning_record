[TOC]

## 0. 简介

*目的：利用内置的小乌龟程序，做一个实操，强化通信应用*



## 1. 话题发布

1. 通过计算图结合`ros`命令获取话题与消息信息。
2. 编码实现运动控制节点。
3. 启动 `roscore`、`turtlesim_node` 以及自定义的控制节点，查看运行结果。



### 1. 话题与消息获取

*提前打开小乌龟案例*

**1. 话题获取`/turtle1/cmd_vel`**

通过计算图查看话题，启动计算图：

```shell
rqt_graph
```

或者利用`rostopic list`，查看所有话题



**2. 消息获取**

**获取消息类型**`geometry_msgs/Twist`

```
rostopic type /turtle1/cmd_vel
```

**获取消息格式**

```shell
rosmsg info geometry_msgs/Twist
```

**得到消息的数据类型如下**

```shell
geometry_msgs/Vector3 linear
  float64 x
  float64 y
  float64 z
geometry_msgs/Vector3 angular
  float64 x
  float64 y
  float64 z
```



### 2. 实现发布节点

*注意这里需要包含依赖包`geometry_msgs`*

注：这里我没有在`CMakeLists.txt`中设置，但也可以编译成功

```cmake
# 设置构建所需要的软件包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)
# 在这里没有添加geometry_msgs
```



**`C++`方案**

```c++
/*
    编写 ROS 节点，控制小乌龟画圆

    准备工作:
        1.获取topic(已知: /turtle1/cmd_vel)
        2.获取消息类型(已知: geometry_msgs/Twist)
        3.运行前，注意先启动 turtlesim_node 节点

    实现流程:
        1.包含头文件
        2.初始化 ROS 节点
        3.创建发布者对象
        4.循环发布运动控制消息
*/

#include "ros/ros.h"
#include "geometry_msgs/Twist.h"

int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "");

    // 初始化ROS节点
    ros::init(argc, argv, "control");

    // 创建ROS句柄
    ros::NodeHandle nh;

    // 创建发布者对象，第一个参数为话题名
    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel",1000);

    // 组织需要发送的消息
    geometry_msgs::Twist msg;
    msg.linear.x = 1.0;
    msg.linear.y = 0.0;
    msg.linear.z = 0.0;

    msg.angular.x = 0.0;
    msg.angular.y = 0.0;
    msg.angular.z = 2.0;

    // 设置发送频率
    ros::Rate r(10);
    // 循环发送
    while (ros::ok())
    {
        pub.publish(msg);

        ros::spinOnce();
    }

    return 0;
}
```



### 3. 编辑配置文件，编译并执行

添加配置如下

```cmake
add_executable(geometry_control src/geometry_pub.cpp)

target_link_libraries(geometry_control
  ${catkin_LIBRARIES}
)
```

编译

```shell
cd first_project
catkin_make
```

新启一个终端，执行

```shell
source ./devel/setup.bash
rosrun first_package geometry_control
```



## 2. 话题订阅

**实现流程:**

1. 通过`ros`命令获取话题与消息信息。
2. 编码实现位姿获取节点。
3. 启动 `roscore`、`turtlesim_node` 、控制节点以及位姿订阅节点，控制乌龟运动并输出乌龟的位姿。



### 1. 话题与消息获取

具体操作参考：[话题发布](#1. 话题与消息获取)

获取到话题：`/turtle1/pose`

消息类型：`turtlesim/Pose`

消息格式：

```
float32 x
float32 y
float32 theta
float32 linear_velocity
float32 angular_velocity
```



### 2. 实现订阅节点

`C++`版本

```c++
/*  
    订阅小乌龟的位姿: 时时获取小乌龟在窗体中的坐标并打印
    准备工作:
        1.获取话题名称 /turtle1/pose
        2.获取消息类型 turtlesim/Pose
        3.运行前启动 turtlesim_node 与 turtle_teleop_key 节点

    实现流程:
        1.包含头文件
        2.初始化 ROS 节点
        3.创建 ROS 句柄
        4.创建订阅者对象
        5.回调函数处理订阅的数据
        6.spin
*/

#include "ros/ros.h"
#include "turtlesim/Pose.h"


void doPose(const turtlesim::Pose::ConstPtr& p)
{
    ROS_INFO("乌龟位姿信息:x=%.2f,y=%.2f,theta=%.2f,lv=%.2f,av=%.2f",
        p->x,p->y,p->theta,p->linear_velocity,p->angular_velocity
    );
}

int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "");

    // 初始化节点
    ros::init(argc, argv, "sub_pose");
    // 创建句柄
    ros::NodeHandle nh;
    // 创建订阅者对象
    ros::Subscriber sub = nh.subscribe<turtlesim::Pose>("/turtle1/pose", 1000, doPose);
    
    ros::spin();

    return 0;
}
```



### 3. 编辑配置文件，编译并执行

添加配置：

```cmake
add_executable(pose_sub src/geometry_sub.cpp)

target_link_libraries(pose_sub
  ${catkin_LIBRARIES}
)
```





编译和执行步骤参考之前的记录，无特别之处。



## 3. 服务调用

**实现流程:**

1. 通过`ros`命令获取服务与服务消息信息。
2. 编码实现服务请求节点。
3. 启动 `roscore`、`turtlesim_node` 、乌龟生成节点，生成新的乌龟。



### 1. 服务名称与服务消息获取

**获取服务：**`/spawn`

```
rosservice list
```

**获取消息类型：**`turtlesim/Spawn`

```
rosservice type /spawn
```

**获取消息格式：**

```
rossrv info turtlesim/Spawn
```

**响应结果：**

```
float32 x
float32 y
float32 theta
string name
---
string name
```



### 2. 实现服务节点

```c++
/*
    生成一只小乌龟
    准备工作:
        1.服务话题 /spawn
        2.服务消息类型 turtlesim/Spawn
        3.运行前先启动 turtlesim_node 节点

    实现流程:
        1.包含头文件
          需要包含 turtlesim 包下资源，注意在 package.xml 配置
        2.初始化 ros 节点
        3.创建 ros 句柄
        4.创建 service 客户端
        5.等待服务启动
        6.发送请求
        7.处理响应

*/

#include "ros/ros.h"
#include "turtlesim/Spawn.h"

int main (int argc, char* argv[])
{
    setlocale(LC_ALL, "");

    // 初始化节点
    ros::init(argc, argv, "set_turtle");

    // 创建句柄
    ros::NodeHandle nh;

    // 创建service客户端，对应服务名
    ros::ServiceClient client = nh.serviceClient<turtlesim::Spawn>("/spawn");

    // 等待服务启动
    ros::service::waitForService("/spawn");
    // client.waitForExistence();

    // 组织请求
    turtlesim::Spawn spawn;
    spawn.request.x = 1.0;
    spawn.request.y = 1.0;
    spawn.request.theta = 1.57;
    spawn.request.name = "my_turtle";

    // 发送请求
    bool flag = client.call(spawn);

    // 处理响应结果
    if (flag)
    {
        ROS_INFO("新的乌龟生成成功，名字是%s", spawn.response.name.c_str());
    }
    else
    {
        ROS_INFO("新乌龟生成失败");
    }

    return 0;
}
```



### 3. 编辑配置文件，编译并执行

在`CMakeLists.txt`中添加：

```cmake
add_executable(Add_MyTurtle src/set_client.cpp)

target_link_libraries(Add_MyTurtle
  ${catkin_LIBRARIES}
)
```



编译和执行步骤参考之前的记录，无特别之处。



## 4. 参数设置

