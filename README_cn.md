# MultiTurnCIR

基于 FastAPI 的多轮图像检索（Multiturn Fashion Retrieval）演示项目。

## 项目与来源说明

- 项目 GitHub：[https://github.com/Tmq244/MultiTurnCIR](https://github.com/Tmq244/MultiTurnCIR)
- 本项目使用模型：**CFIR**
  - 论文（SIGIR）：*Conversational Fashion Image Retrieval via Multiturn Natural Language Feedback*
  - 论文链接：[https://arxiv.org/abs/2106.04128](https://arxiv.org/abs/2106.04128)
- 训练代码仓库：[https://github.com/Tmq244/CFIR](https://github.com/Tmq244/CFIR)
- 使用数据库：[Multi-Turn FashionIQ](https://github.com/yfyuan01/MultiturnFashionRetrieval/tree/master)

用户可以先选择一张参考图，再输入多轮 `modified text`（例如“改成无袖、颜色更亮”），系统会在图库中持续检索更符合描述的目标图像。

## 1. 项目目录结构

```text
MultiTurnCIR/
├─ best_model.pth                  # 模型权重
├─ requirements.txt                # Python 依赖
├─ attr/                           # 属性标注数据
├─ cache/                          # 检索索引缓存（全量/基准子集）
│  ├─ cache_all/
│  ├─ cache_bench_200/
│  ├─ cache_bench_400/
│  └─ cache_bench_1000/
├─ data/                           # 训练/验证数据集
├─ images/                         # 检索展示图片
└─ src/
   ├─ run.py                       # 启动入口（解析 host/port/device 等参数）
   ├─ app/
   │  ├─ main.py                   # FastAPI 应用与路由
   │  ├─ config.py                 # 配置读取（环境变量/路径）
   │  ├─ model_service.py          # 模型加载与向量生成
   │  ├─ retrieval_service.py      # 索引构建与向量检索
   │  ├─ session_service.py        # 多轮会话状态管理
   │  ├─ schemas.py                # API 数据模型
   │  ├─ static/                   # 前端静态资源（JS/CSS）
   │  └─ templates/                # 页面模板
   ├─ Model/                       # 模型结构定义
   └─ preprocess/                  # 数据预处理/训练相关代码
```

## 2. 环境准备

建议使用 Python 3.10+。

```bash
python -m venv .venv
```

Linux 下激活虚拟环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果服务器没有 CUDA/GPU，建议按以下顺序安装：

1. 先安装 CPU 版 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. 再安装其余依赖：

```bash
pip install -r requirements.txt
```

说明：按当前 `requirements.txt`（`torch>=2.0.0`、`torchvision>=0.15.0`），只要版本满足约束，后续安装通常会保留已安装的 CPU 版本，不会重复替换。

下载所需资源：

- `cache` 可从以下链接下载：[here](https://drive.google.com/file/d/1W7h2lpgToHAMlTJ2y4PGsct3u9jPV5hY/view?usp=sharing)
- `images` 可从以下链接下载：[here](https://drive.google.com/file/d/1pivWpO3_vpMLhySmQc9w53i9Tp0ib1lg/view)
- `best_model.pth` 可从以下链接下载：[here](https://huggingface.co/Tmq244/CFIR/tree/main)

下载后请分别放到项目根目录下的：

- `cache/`
- `images/`
- `best_model.pth`

## 3. 启动方式

项目推荐启动命令：

```bash
python src/run.py --device cpu --host 127.0.0.1 --port 8001
```

启动后访问：

- `http://127.0.0.1:8001`

可选参数：

- `--device`：`cpu` 或 `cuda`，默认 `cpu`
- `--host`：服务监听地址，默认 `127.0.0.1`
- `--port`：服务端口，默认 `8001`
- `--index-limit`：限制索引规模（例如 `200`、`400`、`1000`），未传时使用全量缓存

说明：

- 不传 `--index-limit` 时，默认使用 `cache/cache_all`
- 传入 `--index-limit N` 时，使用 `cache/cache_bench_N`

## 4. 页面使用流程

打开首页后，按以下步骤操作：

1. 选择首轮参考图（可在输入框中直接输入图片编码）。
2. 在当前轮输入 `modified text`，点击 `Retrieve`。
3. 在结果中点击一张图，作为下一轮参考图。
4. 重复输入新一轮文本进行多轮检索。
5. 点击 `Reset Session` 可重置会话。

## 5. 主要 API

- `GET /api/health`：服务健康状态、模型加载状态、索引大小
- `GET /api/gallery`：获取图库样本
- `GET /api/reference/{image_id}`：检查参考图是否存在
- `POST /api/session/new`：创建会话
- `POST /api/session/{session_id}/retrieve`：执行一轮检索
- `POST /api/session/{session_id}/reset`：重置会话

示例：创建会话

```bash
curl -X POST "http://127.0.0.1:8001/api/session/new" \
  -H "Content-Type: application/json" \
  -d "{\"reference_id\":\"B000SYGLHE\"}"
```

示例：执行检索

```bash
curl -X POST "http://127.0.0.1:8001/api/session/<session_id>/retrieve" \
  -H "Content-Type: application/json" \
  -d "{\"modified_text\":\"make it sleeveless and brighter\",\"top_k\":10}"
```

## 6. 常见问题

- 模型加载慢：首次启动会进行模型与索引初始化，耗时取决于设备和索引规模。
- 图片无法显示：确认 `images/` 目录存在并且包含对应图片文件。
- GPU 未生效：启动时使用 `--device cuda`，并确认本机 PyTorch CUDA 环境可用。
