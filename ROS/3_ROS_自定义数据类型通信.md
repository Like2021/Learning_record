# 1. 自定义话题通信数据类型

**流程：**

1. 创建`msg`文件，包含有自定义的数据类型
2. 编辑配置文件
3. 编译生成可以被`Python/C++`调用的中间文件
4. 编写发布方和订阅方实现
5. 编辑配置文件，并编译执行



## 1. 定义`msg`文件

在功能包下新建`msg`目录，并新建一个`Person.msg`文件，添加如下内容，定义数据

```
string name
uint16 age
float64 height
```



## 2. 编辑配置文件

`package.xml`中添加编译依赖并执行依赖

```xml
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
```

`CMakeLists.txt`编辑`msg`相关配置

```cmake
# 需要加入 message_generation,必须有 std_msgs
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

## 配置 msg 源文件
add_message_files(
  FILES
  Person.msg
)

# 生成消息时依赖于 std_msgs
generate_messages(
  DEPENDENCIES
  std_msgs
)

#执行时依赖
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES demo02_talker_listener
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)
```



## 3. 编译

```shell
cd first_project
# cd ${工作空间名}
catkin_make
```

编译成功之后会在`.../first_project/devel/include/first_package/`下生成相应头文件

为方便在`VSCode`中编写程序，先将这个头文件地址添加到配置`c_cpp_properties.json`中：

```json
"${workspaceFolder}/devel/include/**"
```



## 4. 编写发布方和订阅方代码

### 1. 发布方

```c++
#include "ros/ros.h"
#include "first_package/Person.h"


int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "");

    // 1.初始化ROS节点
    ros::init(argc, argv, "talker_person");

    // 2.创建ROS句柄
    ros::NodeHandle nh;

    // 3.创建发布者对象，第一个参数为话题名，第二个参数为message队列的大小
    ros::Publisher pub = nh.advertise<first_package::Person>("chatter_person", 1000);

    // ros::Duration(3).sleep();

    // 4.组织被发布的消息，编写发布逻辑并发布消息
    first_package::Person p;
    p.name = "Like";
    p.age = 23;
    p.height = 1.75;

    // 发布逻辑
    ros::Rate r(1);

    while (ros::ok())
    {
        pub.publish(p);
        p.age += 1;
        ROS_INFO("我叫: %s, 今年%d岁, 身高是%.2f米", p.name.c_str(), p.age, p.height);

        r.sleep();
        ros::spinOnce();
    }
    
    return 0;
}
```

### 2. 订阅方

```c++
#include "ros/ros.h"
#include "first_package/Person.h"

// 创建回调函数
void doPerson(const first_package::Person::ConstPtr& p)
{
    ROS_INFO("我订阅了: 姓名是%s, 今年%d岁, 身高%.2f米", p->name.c_str(), p->age, p->height);

}

int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "");

    // 1.初始化ROS节点
    ros::init(argc, argv, "listener_person");

    // 2.创建ROS句柄
    ros::NodeHandle nh;

    // 3.创建订阅者对象
    ros::Subscriber sub = nh.subscribe<first_package::Person>("chatter_person", 10, doPerson);

    ros::spin();
    
    return 0;
}
```



## 5. 配置`CMakeLists.txt`，编译并执行

**配置文件添加：**

```cmake
add_executable(person_talker src/msg_pub.cpp)
add_executable(person_listener src/msg_sub.cpp)

add_dependencies(person_talker ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(person_listener ${PROJECT_NAME}_generate_messages_cpp)

target_link_libraries(person_talker
  ${catkin_LIBRARIES}
)
target_link_libraries(person_listener
  ${catkin_LIBRARIES}
)
```

**编译：**

```shell
cd first_project
catkin_make
```

**执行：**

```shell
# 1.启动roscore
roscore

# 2.新启一个终端，打开发布方节点
cd first_project
source ./devel/setup.bash
rosrun first_package person_talker
# rosrun ${package_name} ${可执行文件}

# 3.同样的方式，打开订阅方节点
```



# 2. 自定义服务通信数据类型

## 0. 简介

服务通信也是ROS中一种极其常用的通信模式，服务通信是基于**请求响应**模式的，是一种应答机制。也即: 一个节点A向另一个节点B发送请求，B接收处理请求并产生响应结果返回给A。比如如下场景:

> 机器人巡逻过程中，控制系统分析传感器数据发现可疑物体或人... 此时需要拍摄照片并留存。

在上述场景中，就使用到了服务通信。

- 一个节点需要向相机节点发送拍照请求，相机节点处理请求，并返回处理结果

与上述应用类似的，服务通信更适用于对时时性有要求、具有一定逻辑处理的应用场景。



#### **作用**

用于偶然的、对时时性有要求、有一定逻辑处理需求的数据传输场景。



**流程：**

1. 创建`srv`文件，包含有自定义的数据类型
2. 编辑配置文件
3. 编译生成可以被`Python/C++`调用的中间文件
4. 编写发布方和订阅方实现
5. 编辑配置文件，并编译执行



## 1. 定义`srv`文件

服务通信中，数据分为两部分，请求与响应，在`srv`文件中请求和响应使用`---`分割，具体实现如下

在功能包下新建`srv`文件夹，并在其中新建`nums.srv`文件，内容如下

```
# 客户端请求时发送的两个数字
int32 num1
int32 num2
---
# 服务器响应发送的数据
int32 sum
```



## 2. 编辑配置文件

*这里与自定义话题通信的操作类似*

需要编辑的是功能包下的`package.xml`和`CMakeLists.txt`，具体如下

1. 首先在`package.xml`中添加编译依赖和执行依赖

```xml
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
<!-- 
exce_depend 以前对应的是 run_depend 现在非法
-->
```

2. 在`CMakeLists.txt`中添加`srv`相关配置

```cmake
# 设置构建所需要的软件包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)
# 配置 srv 源文件
add_service_files(
  FILES
  nums.srv
)
# 生成消息时依赖于 std_msgs
generate_messages(
  DEPENDENCIES
  std_msgs
)
# 运行时依赖
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES first_package
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)
```



## 3. 编译

见上方[自定义话题编译](#3. 编译)



## 4. 编写服务端和客户端代码

### 1. 服务端

```c++
// 1.包含头文件
#include "ros/ros.h"
#include "first_package/nums.h"

// bool 返回值用于标志是否处理成功
bool doReq(first_package::nums::Request& req, first_package::nums::Response& resp)
{
    int num1 = req.num1;
    int num2 = req.num2;

    ROS_INFO("服务器接收到的请求数据为:num1 = %d, num2 = %d",num1, num2);

    // 逻辑处理
    if (num1 < 0 || num2 < 0)
    {
        ROS_ERROR("提交的数据异常: 数据不可以为负数");
        return false;
    }

    // 如果没有ERROR，那就相加并将结果赋值给resp
    resp.sum = num1 + num2;

    return true;
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL,"");
    // 2.初始化 ROS 节点
    ros::init(argc,argv,"AddInts_Server");
    // 3.创建 ROS 句柄
    ros::NodeHandle nh;
    // 4.创建 服务 对象
    ros::ServiceServer server = nh.advertiseService("AddInts", doReq);
    ROS_INFO("服务已经启动....");
    //     5.回调函数处理请求并产生响应
    //     6.由于请求有多个，需要调用 ros::spin()
    ros::spin();
    return 0;
}
```

### 2. 客户端

```c++
#include "ros/ros.h"
#include "first_package/nums.h"


int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "");

    // 调用时动态传值，argc默认为1，即可执行文件的文件名
    if (argc != 3)
    {
        ROS_ERROR("请传递两个参数");
        return 1;
    }

    // 1.初始化节点
    ros::init(argc, argv, "AddInts_Client");
    // 2.创建句柄
    ros::NodeHandle nh;
    // 3.创建 客户端 对象
    ros::ServiceClient client = nh.serviceClient<first_package::nums>("AddInts");
    // 4.等待服务启动成功
    // 方式1
    // ros::service::waitForService("AddInts");
    // 方式2
    client.waitForExistence();
    // 5.组织请求数据
    first_package::nums nums;
    nums.request.num1 = atoi(argv[1]);
    nums.request.num2 = atoi(argv[2]);
    // 6.发送请求，返回bool值，标记是否成功
    bool flag = client.call(nums);
    // 7.处理响应
    if (flag)
    {
        ROS_INFO("请求正常处理，响应结果: %d", nums.response.sum);
    }
    else
    {
        ROS_ERROR("请求失败...");
        return 1;
    }
    
    return 0;
}
```



## 5. 配置`CMakeLists.txt`，编译并执行

**添加如下**

```cmake
add_executable(AddInts_Server src/srv_server.cpp)
add_executable(AddInts_Client src/srv_client.cpp)

add_dependencies(AddInts_Server ${PROJECT_NAME}_gencpp)
add_dependencies(AddInts_Client ${PROJECT_NAME}_gencpp)

target_link_libraries(AddInts_Server
  ${catkin_LIBRARIES}
)
target_link_libraries(AddInts_Client
  ${catkin_LIBRARIES}
)
```

**编译和执行步骤如之前一致**

