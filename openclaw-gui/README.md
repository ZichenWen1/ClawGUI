# OpenGUI-Deployment

1. [项目简介](#1-项目简介)
2. [核心功能](#2-核心功能)
3. [系统架构](#3-系统架构)
4. [目录结构](#4-目录结构)
5. [支持的模型](#5-支持的模型)
6. [支持的设备与平台](#6-支持的设备与平台)
7. [支持的应用程序](#7-支持的应用程序)
8. [环境要求与安装](#8-环境要求与安装)
9. [模型部署](#9-模型部署)
10. [使用方法](#10-使用方法)
    - [命令行界面 (CLI)](#101-命令行界面-cli)
    - [Web 图形界面](#102-web-图形界面)
12. [支持的动作类型](#12-支持的动作类型)
13. [记忆系统](#13-记忆系统)
14. [回调与手动接管](#14-回调与手动接管)
15. [常见问题与排错](#15-常见问题与排错)

---

## 1. 项目简介

**Omni-GUI** 是开源的智能手机自动化框架，允许 AI 视觉语言模型（VLM）自主理解手机屏幕内容、规划操作步骤、并直接控制手机完成复杂任务。

用户只需用自然语言描述任务（如"帮我在淘宝上搜索无线耳机并加入购物车"），AI 即可自动完成一系列点击、滑动、输入等操作。

### 核心理念

```
用户自然语言指令 → VLM理解屏幕 → 规划动作 → 执行 → 截图 → 循环直到任务完成
```

### 项目亮点

- **多平台支持**：Android、鸿蒙 OS（HarmonyOS）、iOS
- **多模型适配**：AutoGLM、Qwen VL、UI-TARS、GLM-4V、MAI-UI
- **多接口形式**：命令行、Web UI、Python API
- **个性化记忆**：学习并记住用户偏好（联系人、常用App等）
- **OpenAI 兼容**：连接任意支持 OpenAI 格式的模型服务

---

## 2. 核心功能

### 2.1 视觉理解与自主操作

AI 每步都会截取当前屏幕，将图片传送给视觉语言模型，由模型决定下一步操作（点击、滑动、输入文字等），形成闭环执行。

### 2.2 多平台设备控制

| 平台 | 工具 | 连接方式 |
|------|------|----------|
| Android | ADB（Android Debug Bridge） | USB / WiFi / 远程 |
| 鸿蒙 OS | HDC（Huawei Device Connection） | USB |
| iOS | WebDriverAgent / XCTest | WiFi |

### 2.3 丰富的交互动作

支持以下设备操作：

- `Tap`：点击指定坐标
- `Double Tap`：双击
- `Long Press`：长按
- `Swipe`：滑动（上下左右）
- `Type`：输入文本
- `Launch`：启动应用
- `Back`：返回上一页
- `Home`：回到主页
- `Wait`：等待页面加载
- `Take_over`：请求人工接管（适用于验证码、登录等场景）
- `Interact`：让用户选择选项
- `Note`：记录页面内容
- `Call_API`：总结页面内容
- `finish`：任务完成

### 2.4 个性化记忆系统

- 自动提取并存储用户联系人、App 偏好、使用习惯
- 下次执行相似任务时，自动填充已知信息（如联系人姓名）
- 支持多用户隔离存储

### 2.5 敏感操作保护

框架内置确认回调机制，对于涉及支付、消息发送等敏感操作，会先请求用户确认，避免误操作。

### 2.6 流式输出

模型响应支持流式传输，CLI 和 Web UI 均可实时展示 AI 的"思考过程"。

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户接口层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  main.py     │  │  webui.py    │  │  Python API      │  │
│  │  (CLI)       │  │  (Gradio)    │  │  (phone_agent)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      智能体核心层                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PhoneAgent / IOSPhoneAgent                          │   │
│  │  - 任务调度  - 步骤循环  - 状态管理                   │   │
│  └──────────────────────────────────────────────────────┘   │
│       │                    │                    │            │
│  ┌────────────┐   ┌─────────────────┐  ┌──────────────┐    │
│  │ ModelClient│   │  ActionHandler  │  │MemoryManager │    │
│  │ (AI推理)   │   │ (动作执行)      │  │ (记忆系统)   │    │
│  └────────────┘   └─────────────────┘  └──────────────┘    │
│       │                    │                                 │
│  ┌────────────┐   ┌─────────────────┐                       │
│  │  模型适配器 │   │  设备工厂       │                       │
│  │  (Adapters)│   │  (DeviceFactory)│                       │
│  └────────────┘   └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      设备控制层                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │ ADB 模块   │  │ HDC 模块   │  │ XCTest 模块 (iOS)  │    │
│  │(Android)  │  │(鸿蒙)      │  │                    │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      物理设备层                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │ Android    │  │ HarmonyOS  │  │ iOS                │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

```
1. 用户输入任务描述
2. Agent 截取当前屏幕（Base64 编码图像）
3. ModelClient 将截图 + 系统提示词 + 历史对话发给 VLM
4. VLM 返回下一步动作（如 do(action="Tap", x=300, y=500)）
5. ActionHandler 解析并执行动作
6. 重复 2-5，直到模型输出 finish() 或达到最大步数
```

---

## 4. 目录结构

```
Open-AutoGLM-main/
├── main.py                      # 命令行入口
├── webui.py                     # Gradio Web 界面入口
├── ios.py                       # iOS 专用入口
├── setup.py                     # 包安装配置
├── requirements.txt             # Python 依赖
├── README.md                    # 中文文档
├── README_en.md                 # 英文文档
├── README_coding_agent.md       # 为 AI 助手准备的自动化部署指南
│
├── phone_agent/                 # 核心包
│   ├── __init__.py
│   ├── agent.py                # PhoneAgent 主类（Android/鸿蒙）
│   ├── agent_ios.py            # IOSPhoneAgent 类
│   ├── device_factory.py       # 设备类型工厂
│   │
│   ├── config/                 # 配置与提示词
│   │   ├── apps.py            # Android App 包名映射（50+）
│   │   ├── apps_harmonyos.py  # 鸿蒙 App 映射（60+）
│   │   ├── apps_ios.py        # iOS App 映射
│   │   ├── prompts_zh.py      # 中文系统提示词
│   │   ├── prompts_en.py      # 英文系统提示词
│   │   ├── prompts_glm4v.py   # GLM-4V 专用提示词
│   │   ├── prompts_uitars.py  # UI-TARS 专用提示词
│   │   ├── prompts_qwenvl.py  # Qwen VL 专用提示词
│   │   ├── prompts_maiui.py   # MAI-UI 专用提示词
│   │   ├── i18n.py            # 国际化支持
│   │   └── timing.py          # 动作时序配置
│   │
│   ├── model/                  # 模型客户端
│   │   ├── client.py          # OpenAI 兼容客户端
│   │   └── adapters.py        # 多模型适配器
│   │
│   ├── adb/                    # Android ADB 工具
│   │   ├── connection.py      # 连接管理（USB/WiFi/远程）
│   │   ├── device.py          # 设备控制（点击/滑动等）
│   │   ├── screenshot.py      # 截图
│   │   └── input.py           # 文字输入（ADB Keyboard）
│   │
│   ├── hdc/                    # 鸿蒙 HDC 工具
│   │   ├── connection.py
│   │   ├── device.py
│   │   ├── screenshot.py
│   │   └── input.py
│   │
│   ├── xctest/                 # iOS XCTest 工具
│   │   ├── screenshot.py
│   │   └── input.py
│   │
│   ├── actions/                # 动作处理器
│   │   ├── handler.py         # 基础动作处理器
│   │   ├── handler_ios.py     # iOS 专用
│   │   ├── handler_uitars.py  # UI-TARS 专用
│   │   ├── handler_qwenvl.py  # Qwen VL 专用
│   │   └── handler_maiui.py   # MAI-UI 专用
│   │
│   └── memory/                 # 记忆系统
│       ├── memory_store.py    # 记忆持久化存储
│       └── memory_manager.py  # 记忆提取与管理
│
├── examples/                   # 使用示例
│   ├── basic_usage.py
│   ├── demo_thinking.py
│   └── demo_memory.py
│
├── scripts/                    # 工具脚本
│   ├── check_deployment_cn.py # 中文部署验证
│   └── check_deployment_en.py # 英文部署验证
│
└── memory_db/                  # 记忆数据库（运行时生成）
```

---

## 5. 支持的模型

框架通过适配器模式支持多种视觉语言模型，每种模型有专属的系统提示词和响应解析逻辑。

### 5.1 官方推荐模型

| 模型 | 类型标识 | 说明 |
|------|----------|------|
| **AutoGLM-Phone-9B** | `AUTOGLM` | 智谱AI专为手机控制优化的9B模型，中文最优 |
| **AutoGLM-Phone-9B-Multilingual** | `AUTOGLM` | 多语言版本，支持英文任务 |

**下载地址**：
- Hugging Face：`zai-org/AutoGLM-Phone-9B`
- ModelScope：`ZhipuAI/AutoGLM-Phone-9B`

### 5.2 第三方兼容模型

| 模型 | 类型标识 | 提供商 |
|------|----------|--------|
| **GLM-4.6V-flash** | `GLM4V` | 智谱AI云端 API |
| **GLM-4.1V-9B-thinking** | `GLM4V` | 智谱AI，带推理过程 |
| **Doubao-1.5-UI-TARS** | `UITARS` | 字节跳动 |
| **Qwen2.5-VL** | `QWENVL` | 阿里云通义 |
| **Qwen3-VL** | `QWENVL` | 阿里云通义 |
| **MAI-UI** | `MAIUI` | 阿里云 MAI |

### 5.3 模型推理服务配置

框架通过 **OpenAI 兼容 API** 与模型通信，支持以下推理服务提供商：

| 服务商 | API 地址 | 备注 |
|--------|---------|------|
| 本地 vLLM | `http://localhost:8000/v1` | 推荐，延迟最低 |
| 本地 SGLang | `http://localhost:8000/v1` | 高性能推理 |
| ModelScope | `https://api-inference.modelscope.cn/v1` | 魔塔社区 |
| 智普BigModel | `https://open.bigmodel.cn/api/paas/v4` | 国内访问 |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 国内访问 |
| 火山引擎 | `https://ark.cn-beijing.volces.com/api/v3` | 国内访问 |

---

## 6. 支持的设备与平台

### 6.1 Android

- **连接工具**：ADB（Android Debug Bridge）
- **连接方式**：USB、WiFi（同局域网）、远程网络
- **系统要求**：手机开启开发者模式 + USB 调试
- **输入法**：需安装 ADB Keyboard（框架会自动检测提示）

### 6.2 鸿蒙 OS（HarmonyOS）

- **连接工具**：HDC（Huawei Device Connection）
- **需求**：鸿蒙开发者工具包
- **支持 App**：60+ 鸿蒙原生应用

### 6.3 iOS

- **连接工具**：WebDriverAgent + XCTest
- **需求**：Mac 系统 + Xcode + WebDriverAgent
- **入口文件**：`ios.py`（专用入口）

### 6.4 分辨率适配

框架自动获取设备屏幕分辨率，坐标系基于实际屏幕像素。

---

## 7. 支持的应用程序

### 7.1 Android 应用（50+）

| 分类 | 应用 |
|------|------|
| **社交** | 微信、QQ、微博 |
| **电商** | 淘宝、京东、拼多多、Temu、eBay、Amazon |
| **外卖/餐饮** | 美团、饿了么、大众点评 |
| **出行** | 携程、12306、滴滴、高德地图、Google Maps |
| **视频** | 哔哩哔哩、抖音、TikTok、腾讯视频、爱奇艺 |
| **音乐** | 网易云音乐、QQ音乐、Spotify |
| **效率** | Gmail、Google日历、Google Drive、Joplin |
| **购物** | 小红书 |
| **学习** | Duolingo |
| **健康** | Google Fit |
| **工具** | Chrome、Google Clock、Google Play、Files |

### 7.2 鸿蒙应用（60+）

鸿蒙版本支持更多华为生态应用，包括：华为视频、华为音乐、华为钱包、智慧出行等。

### 7.3 iOS 应用

Safari、邮件、地图、信息等系统应用，以及微信等主流 App。

### 7.4 自定义 App

若需要控制未内置映射的 App，可直接在任务中使用完整包名，或在 `phone_agent/config/apps.py` 中添加映射。

---

## 8. 环境要求与安装

### 8.1 系统要求

- **操作系统**：Linux / macOS / Windows（WSL）
- **Python**：3.10 或更高
- **显卡**（本地部署）：NVIDIA GPU，推荐 24GB+ VRAM

### 8.2 安装步骤

```bash
pip install -r requirements.txt
pip install -e .
```

### 8.3 主要依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `Pillow` | >=12.0.0 | 图像处理 |
| `openai` | >=2.9.0 | OpenAI 兼容 API 客户端 |
| `gradio` | >=4.0.0 | Web 界面 |
| `requests` | >=2.31.0 | HTTP 请求 |
| `numpy` | >=1.24.0 | 数值计算 |

### 8.4 设备工具安装

##### 下载ADB（Android）

下载 [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en)（ADB），即 SDK-Platform-Tools，里面自动包含了ADB应用程序，根据系统选择合适的文件包。

##### 下载HDC（HarmonyOS NEXT版本以上）

从[HarmonyOS SDK](https://developer.huawei.com/consumer/cn/download/)下载

##### **配置环境变量（Windows）**

​	1.右键「此电脑」→「属性」→「高级系统设置」→「环境变量」
​	2.在「系统变量」中找到 Path，点击「编辑」→「新建」
​	3.输入 platform-tools 文件夹的完整路径（例如 C:\platform-tools）
​	4.点击「确定」保存所有窗口。

##### 配置环境变量（MacOS/Linux）

在`Terminal`或者任何命令行工具里

```
# 假设解压后的目录为 ~/Downloads/platform-tools。如果不是请自行调整命令。
export PATH=${PATH}:~/Downloads/platform-tools
```

##### 验证安装

在`Terminal`或者任何命令行工具里输入
![image-20260316150536360](C:\Users\HUAWEI\AppData\Roaming\Typora\typora-user-images\image-20260316150536360.png)

若显示版本信息（如 `Android Debug Bridge version 1.0.41`），则安装成功。

##### 打开USB调试

对于一般的Android移动设备，找到设置-关于手机-系统详细参数界面，连续点击7次~10次版本号栏，就能收到“您已处于开发者模式”的提示，此时你可以在系统的高级设置或其他设置界面中就能找到“开发者选项”功能的入口，即可开启“USB调试”或“ADB调试”，如果是HyperOS系统需要同时打开 "[USB调试(安全设置)](https://github.com/user-attachments/assets/05658b3b-4e00-43f0-87be-400f0ef47736)"。

##### 连接设备

通过数据线连接移动设备和电脑，同时在手机的连接选项中选择“传输文件”。

你可以在终端通过下面的命令来测试你的连接是否成功: `/path/to/adb devices`。如果输出的结果显示你的设备列表不为空，则说明连接成功。

##### 注意事项

1. 如果你是用的是MacOS或者Linux，请先为 ADB 开启权限: `sudo chmod +x /path/to/adb`。
2. `/path/to/adb`在Windows电脑上将是`xx/xx/adb.exe`的文件格式，而在MacOS或者Linux则是`xx/xx/adb`的文件格式。

##### 在你的移动设备上安装 ADB 键盘

ADBKeyBoard 是一款专为 Android 自动化测试设计的虚拟键盘工具，通过 ADB 命令实现文本输入功能。该工具解决了 Android 系统内置 input 命令无法发送Unicode 字符的痛点，让中文输入和特殊字符处理变得简单高效。

下载 ADB 键盘的 [apk](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk) 安装包，在设备上点击该 apk 来安装，在系统设置中将默认输入法切换为 “ADB Keyboard” 。如果在手机下边栏看到 “ADB Keyboard(on)” 则说明已成功启动 ADB 键盘。

##### 对于Android设备

确定已安装ADB并使用**USB数据线**连接设备：

```
# 检查已连接的设备
adb devices

# 输出结果应显示你的设备，如：
# List of devices attached
# emulator-5554   device
```

##### 对于鸿蒙设备

确定已安装HDC并使用**USB数据线**连接设备：

```
# 检查已连接的设备
hdc list targets

# 输出结果应显示你的设备，如：
# 7001005458323933328a01bce01c2500
```

### 8.5 验证部署

```bash
# 中文部署验证
python scripts/check_deployment_cn.py

# 英文部署验证
python scripts/check_deployment_en.py
```

---

## 9. 模型部署

### 9.1 使用 vLLM（推荐）

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --served-model-name autoglm-phone-9b \
  --allowed-local-media-path / \
  --mm-encoder-tp-mode data \
  --mm_processor_cache_type shm \
  --mm_processor_kwargs "{\"max_pixels\":5000000}" \
  --max-model-len 50960 \
  --chat-template-content-format string \
  --limit-mm-per-prompt "{\"image\":10}" \
  --tensor-parallel-size 1 \
  --model /home/model/zai-org/AutoGLM-Phone-9B \
  --port 6007
```

### 9.2 使用 SGLang

```bash
python3 -m sglang.launch_server --model-path  /home/model/zai-org/AutoGLM-Phone-9B \
        --served-model-name autoglm-phone-9b  \
        --context-length 25480  \
        --mm-enable-dp-encoder   \
        --mm-process-config '{"image":{"max_pixels":5000000}}'  \
        --port 8000
```

### 9.3 验证模型服务

```bash
curl http://localhost:8000/v1/models
```

### 9.4 使用云端 API（无需本地 GPU）

```bash
# 使用智谱 BigModel 服务
python main.py \
  --base-url https://open.bigmodel.cn/api/paas/v4 \
  --model "autoglm-phone" \
  --api-key "autoglm-phone" \
  "打开微信查看最新消息"
```

---

## 10. 使用方法

### 10.1 命令行界面 (CLI)

**基本格式**：

```bash
python main.py [选项] "任务描述"
```

**常用参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url` | `http://localhost:8000/v1` | 模型 API 地址 |
| `--model` | `autoglm-phone-9b` | 模型名称 |
| `--api-key` | `EMPTY` | API 密钥 |
| `--device-type` | `adb` | 设备类型：`adb` / `hdc` |
| `--device-id` | 自动检测 | 设备 ID（多设备时指定） |
| `--lang` | `cn` | 语言：`cn` / `en` |
| `--max-steps` | `100` | 最大执行步数 |
| `--verbose` | `false` | 显示详细调试信息 |
| `--list-apps` | - | 列出已安装应用 |
| `--model-type` | `autoglm` | 模型类型适配器 |
| `--thinking` | - | 开启思维链模式（需模型支持） |
| `--memory-dir` | `memory_db` | 记忆数据库目录 |
| `--no-memory` | - | 禁用记忆系统 |

**使用示例**：

```bash
# 1. 最简用法（本地模型）
python main.py "打开微信给张三发消息说我晚点到"

# 2. 指定云端模型
python main.py \
  --base-url https://api.z.ai/api/paas/v4 \
  --model glm-4.6v-flash \
  --api-key sk-xxx \
  "在淘宝搜索蓝牙耳机"

# 3. 使用鸿蒙设备
python main.py --device-type hdc "打开华为视频"

# 4. 指定特定设备（多设备场景）
python main.py --device-id "192.168.1.100:5555" "查看今日天气"

# 5. WiFi 远程控制
python main.py --device-id "10.0.0.1:5555" "在京东下单"

# 6. 使用英文提示词
python main.py --lang en "Open WeChat and check messages"

# 7. 开启思维链（使用 GLM-4.1V-thinking）
python main.py \
  --model glm-4.1v-9b-thinking \
  --thinking \
  "帮我在美团订一份外卖"

# 8. 使用 UI-TARS 模型
python main.py \
  --model-type uitars \
  --model doubao-1.5-ui-tars \
  "打开设置调节亮度"

# 9. 使用 Qwen VL 模型
python main.py \
  --model-type qwenvl \
  --model qwen2.5-vl-7b-instruct \
  "打开相册查看最新照片"

# 10. 交互式模式（不指定任务，进入循环）
python main.py

# 11. 列出已安装应用
python main.py --list-apps

# 12. 禁用记忆系统
python main.py --no-memory "查一下天气"
```

### 10.2 Web 图形界面

```bash
python webui.py
```

默认在 `http://localhost:7860` 打开，提供以下功能：

- **设备管理面板**：连接/断开设备、查看设备状态
- **任务执行**：输入任务描述，点击运行，实时查看截图和 AI 思考
- **系统检测**：一键检测 ADB/模型服务/设备连接状态
- **手动接管**：遇到验证码或需人工干预时，切换手动操作模式
- **记忆管理**：查看/编辑/清除记忆数据
- **配置面板**：图形化设置模型参数

---

## 11. 支持的动作类型

模型每步输出的动作格式如下：

### 11.1 基础动作

| 动作 | 格式 | 说明 |
|------|------|------|
| 点击 | `do(action="Tap", x=300, y=500)` | 点击屏幕坐标 |
| 双击 | `do(action="Double Tap", x=300, y=500)` | 双击 |
| 长按 | `do(action="Long Press", x=300, y=500)` | 长按 |
| 滑动 | `do(action="Swipe", x1=300, y1=800, x2=300, y2=200)` | 手势滑动 |
| 输入 | `do(action="Type", text="你好")` | 输入文字 |
| 输入联系人 | `do(action="Type_Name", text="张三")` | 输入人名 |
| 启动 App | `do(action="Launch", app="微信")` | 打开应用 |
| 返回 | `do(action="Back")` | 系统返回键 |
| 主页 | `do(action="Home")` | 系统主页键 |
| 等待 | `do(action="Wait")` | 等待页面加载 |

### 11.2 特殊动作

| 动作 | 格式 | 说明 |
|------|------|------|
| 请求接管 | `do(action="Take_over", message="需要输入验证码")` | 请求人工介入 |
| 交互选择 | `do(action="Interact", message="请选择联系人")` | 让用户做选择 |
| 记录内容 | `do(action="Note", text="页面显示余额为100元")` | 记录关键信息 |
| 内容总结 | `do(action="Call_API", content="...")` | 调用 API 总结内容 |

### 11.3 任务结束

```
finish(message="已成功发送消息给张三")
```

---

## 12. 记忆系统

### 12.1 工作原理

记忆系统在每次任务完成后，从对话历史中提取有价值的信息并持久化存储。下次执行相似任务时，自动将相关记忆注入上下文。

### 12.2 记忆类型

| 类型 | 示例 |
|------|------|
| `contact` | 张三的微信名是"小张" |
| `app` | 用户常用淘宝购物 |
| `preference` | 用户喜欢每天早上查看新闻 |
| `address` | 家庭地址：XX市XX路XX号 |
| `habit` | 每周五下午下单外卖 |

### 12.3 自动提取模式

记忆系统能自动识别对话中的关键信息：

```
"给张三发微信" → 提取联系人：张三
"打开淘宝"     → 记录常用App：淘宝
"发到家里"     → 提示填写家庭地址
```

### 12.4 存储格式

记忆以 JSON 格式存储在 `memory_db/` 目录下：

```
memory_db/
└── {user_id}/
    └── memories.json
```

```json
[
  {
    "id": "uuid",
    "type": "contact",
    "content": "张三的微信昵称是大张",
    "created_at": "2025-01-01T10:00:00",
    "used_count": 5
  }
]
```

---

## 13. 回调与手动接管

### 13.1 确认回调

当模型决定执行敏感操作（如发送支付、删除数据等）时触发：

```python
def my_confirm(message: str) -> bool:
    # message 是模型描述的即将执行的操作
    print(f"准备执行：{message}")
    return True  # 返回 True 继续执行，False 终止
```

### 13.2 接管回调

当模型遇到无法自动处理的情况（验证码、登录、人脸识别等）时触发：

```python
def my_takeover(message: str):
    # message 描述需要手动处理的内容
    print(f"请手动操作：{message}")
    input("完成后按回车继续...")
```

### 13.3 步骤回调

每步执行后触发，可用于监控和日志：

```python
def step_callback(step_info: dict):
    print(f"第{step_info['step']}步：{step_info['action']}")
    # step_info 包含 step, action, screenshot 等字段

agent = PhoneAgent(step_callback=step_callback)
```

---

## 14. 常见问题与排错

### Q1：找不到已连接的设备

```bash
# 检查 ADB 连接
adb devices

# 如果设备显示为 unauthorized
# 在手机上允许 USB 调试授权弹窗
```

### Q2：文字输入失败

```bash
# 确认 ADB Keyboard 已安装并激活
adb shell ime list -s
# 应看到 com.android.adbkeyboard/.AdbIME

# 手动激活
adb shell ime set com.android.adbkeyboard/.AdbIME
```

### Q3：模型服务连接失败

```bash
# 测试模型 API 是否可用
curl http://localhost:8000/v1/models

# 检查端口是否监听
netstat -tlnp | grep 8000
```

### Q4：模型响应慢/超时

- 检查 GPU 显存是否足够（推荐 24GB+）
- 降低 `max_tokens`（减少每步输出长度）
- 使用 `--max-steps` 限制最大步数

### Q5：App 无法启动

```bash
# 查看已安装 App 列表，确认包名
python main.py --list-apps

# 查看特定 App 包名
adb shell pm list packages | grep wechat
```

### Q6：WiFi 连接不稳定

```bash
# 检查网络连通性
adb connect 192.168.x.x:5555

# 如果断开，重新连接
adb disconnect
adb connect 192.168.x.x:5555
```

### Q7：鸿蒙设备不识别

```bash
# 检查 HDC 工具
hdc list targets

# 确认设备已开启 HDC 调试
```

### Q8：任务步数超出限制

```bash
# 增加最大步数
python main.py --max-steps 200 "复杂的多步任务"
```
