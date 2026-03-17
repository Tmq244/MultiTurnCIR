# MT CIR

A FastAPI-based Multiturn Composed Image Retrieve demo.

## Project & References

- Project GitHub: [https://github.com/Tmq244/MultiTurnCIR](https://github.com/Tmq244/MultiTurnCIR)
- Model used in this project: **CFIR**
  - Paper (SIGIR): *Conversational Fashion Image Retrieval via Multiturn Natural Language Feedback*
  - Paper link: [https://arxiv.org/abs/2106.04128](https://arxiv.org/abs/2106.04128)
- Training code: [https://github.com/Tmq244/CFIR](https://github.com/Tmq244/CFIR)
- Dataset used as database: [Multi-Turn FashionIQ](https://github.com/yfyuan01/MultiturnFashionRetrieval/tree/master)

You can start from a reference image, then submit multiple rounds of `modified text` (for example: "make it sleeveless and brighter"). The system keeps retrieving images that better match the evolving description.

## 1. Project Structure

```text
MultiTurnCIR/
├─ best_model.pth                  # Model checkpoint
├─ requirements.txt                # Python dependencies
├─ attr/                           # Attribute annotation data
├─ cache/                          # Retrieval index caches (full/benchmark subsets)
│  ├─ cache_all/
│  ├─ cache_bench_200/
│  ├─ cache_bench_400/
│  └─ cache_bench_1000/
├─ data/                           # Training/validation datasets
├─ images/                         # Images used in gallery and retrieval results
└─ src/
   ├─ run.py                       # Startup entry (parses host/port/device, etc.)
   ├─ app/
   │  ├─ main.py                   # FastAPI app and routes
   │  ├─ config.py                 # Config loading (env vars/paths)
   │  ├─ model_service.py          # Model loading and query embedding
   │  ├─ retrieval_service.py      # Index building and vector search
   │  ├─ session_service.py        # Multiturn session state management
   │  ├─ schemas.py                # API schemas
   │  ├─ static/                   # Frontend static assets (JS/CSS)
   │  └─ templates/                # HTML templates
   ├─ Model/                       # Model architecture modules
   └─ preprocess/                  # Data preprocessing/training utilities
```

## 2. Environment Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the virtual environment in Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For servers without CUDA/GPU, use this recommended installation order:

1. Install CPU-only PyTorch wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Note: with the current `requirements.txt` (`torch>=2.0.0`, `torchvision>=0.15.0`), pip usually keeps the already installed CPU builds if versions satisfy constraints.

Download required resources:

- `cache` can be downloaded from: [here](https://drive.google.com/file/d/1W7h2lpgToHAMlTJ2y4PGsct3u9jPV5hY/view?usp=sharing)
- `images` can be downloaded from: [here](https://drive.google.com/file/d/1pivWpO3_vpMLhySmQc9w53i9Tp0ib1lg/view)
- `best_model.pth` can be downloaded from: [here](https://huggingface.co/Tmq244/CFIR/tree/main)

After downloading, place them in the project root as:

- `cache/`
- `images/`
- `best_model.pth`

## 3. Run the Project

Recommended startup command:

```bash
python src/run.py --device cpu --host 127.0.0.1 --port 8001
```

Then open:

- `http://127.0.0.1:8001`

Optional arguments:

- `--device`: `cpu` or `cuda` (default: `cpu`)
- `--host`: bind host (default: `127.0.0.1`)
- `--port`: bind port (default: `8001`)
- `--index-limit`: limit index size (for example `200`, `400`, `1000`); if omitted, full cache is used

Notes:

- Without `--index-limit`, the app uses `cache/cache_all`
- With `--index-limit N`, the app uses `cache/cache_bench_N`

## 4. Web UI Workflow

After opening the homepage, follow this flow:

1. Select the first-round reference image (or input image ID directly).
2. Enter `modified text` and click `Retrieve`.
3. Click one result image to use it as the next-round reference.
4. Repeat with new text for multiround retrieval.
5. Click `Reset Session` to clear the current session.

## 5. Main APIs

- `GET /api/health`: service health, model status, index size
- `GET /api/gallery`: sample gallery images
- `GET /api/reference/{image_id}`: check whether a reference image exists
- `POST /api/session/new`: create a session
- `POST /api/session/{session_id}/retrieve`: run one retrieval round
- `POST /api/session/{session_id}/reset`: reset a session

Example: create a session

```bash
curl -X POST "http://127.0.0.1:8001/api/session/new" \
  -H "Content-Type: application/json" \
  -d "{\"reference_id\":\"B000SYGLHE\"}"
```

Example: run retrieval

```bash
curl -X POST "http://127.0.0.1:8001/api/session/<session_id>/retrieve" \
  -H "Content-Type: application/json" \
  -d "{\"modified_text\":\"make it sleeveless and brighter\",\"top_k\":10}"
```

## 6. Troubleshooting

- Slow startup: the first run initializes model and index, which may take time.
- Images not displayed: make sure the `images/` directory exists and contains valid image files.
- GPU not used: run with `--device cuda` and confirm your local PyTorch CUDA setup is available.
