# Godot-RAG

一个基于RAG（检索增强生成）的Godot 4.4文档智能问答系统，支持本地向量数据库和MCP服务。

## 🎯 项目简介

Godot-RAG是一个专门为Godot游戏引擎4.4版本文档构建的智能问答系统。通过检索增强生成技术，系统能够从本地向量数据库中快速检索相关文档片段，并结合大语言模型生成准确、上下文相关的答案。

## 📋 环境要求

- Python 3.11.14
- Godot 4.4 文档 (从 [godotengine/godot-docs](https://github.com/godotengine/godot-docs/tree/4.4) 下载)
- 硅基流动API密钥

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/Biggie-Chris/Godot-RAG.git
cd Godot-RAG

# 安装依赖
pip install -r requirements.txt
```

### 2. 获取Godot文档

1. 从 [Godot 4.4文档仓库](https://github.com/godotengine/godot-docs/tree/4.4) 下载文档
2. 解压到项目根目录的 `doc`（自行创建 `doc`文件夹） 文件夹中

### 3. 配置API密钥

编辑 `.env` 文件，设置你的硅基流动API密钥：

```env
OPENAI_API_KEY='你的API密钥'
OPENAI_BASE_URL='https://api.siliconflow.cn/v1'
```

### 4. 预处理文档

```bash
python preprocess.py
```

这个步骤会：
- 解析Godot文档的搜索索引
- 将文档分块处理
- 构建向量数据库

### 5. 运行问答系统

```bash
python main.py
```

启动交互式命令行界面，你可以输入Godot相关问题并获得智能回答。


## 6. 🎮 使用示例

启动系统后，你可以询问各种Godot相关问题：


🎮 Godot RAG 系统启动！输入你的问题，输入 exit 退出。

❓ 你的问题： 如何在Godot中创建一个简单的2D角色控制器？

🔍 正在检索 + 生成回答 ...

💬 回答：

在Godot中创建2D角色控制器，你可以使用以下步骤：

1. 创建一个CharacterBody2D节点
2. 添加CollisionShape2D子节点并设置碰撞形状
3. 在脚本中处理输入和移动逻辑
4. 使用`move_and_slide()`方法实现移动

示例代码：
```gdscript
extends CharacterBody2D

@export var speed: float = 300.0
@export var jump_velocity: float = -400.0

func _physics_process(delta):
    var direction = Input.get_axis("ui_left", "ui_right")
    velocity.x = direction * speed
    
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = jump_velocity
    
    move_and_slide()
```

📚 引用来源：
- doc_id: 123 | 相似度: 0.8567
  ↳ doc/_sources/tutorials/2d/2d_movement.txt
- doc_id: 45 | 相似度: 0.8123
  ↳ doc/_sources/classes/class_characterbody2d.txt


## 🔌 MCP服务

项目还提供了MCP服务，可以集成到其他AI应用中：

```bash
python mcp_server.py
```

MCP服务提供 `rag.query` 工具，支持以下参数：
- `query`: 用户的问题
- `top_k`: 返回的文档数量（默认5）


