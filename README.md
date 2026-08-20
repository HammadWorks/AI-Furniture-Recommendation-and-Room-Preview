# AI Furniture Recommendation and Room Preview

An end-to-end computer vision prototype that recommends furniture from a room photograph. The system combines local multimodal analysis, CLIP-based visual retrieval, and optional generative image editing to turn a room image into ranked furniture suggestions and a visual preview.

## What It Does

Given a room image and a furniture category, the notebook:

1. Analyzes the room's lighting, floor, wall colors, and overall style with a local Gemma model served by Ollama.
2. Converts that analysis into a furniture-oriented text query.
3. Encodes the query and furniture images with CLIP and compares them with cosine similarity.
4. Ranks the closest products and displays the top recommendations in an image grid.
5. Optionally sends the recommendations to an OpenRouter-hosted FLUX model to generate a room preview.

The project also includes a Gradio interface for entering a room image, selecting a furniture type, and viewing results interactively.

## Why This Project Is Interesting

- **Multimodal retrieval:** connects image understanding, language, and image similarity in one workflow.
- **Local-first inference:** room analysis runs through Ollama, which keeps the initial image analysis local and reduces cloud dependence.
- **Resource-aware processing:** CLIP models are loaded lazily, embeddings can be processed in batches, and image caches reduce repeated work.
- **Practical product experience:** the output is both a ranked recommendation list and an optional visualized room concept.

## System Overview

```text
Room image
	|
	v
Gemma via Ollama ----> Room description / furniture query
	|                                 |
	+-------------------------> CLIP text embedding
									  |
Furniture image dataset ------> CLIP image embeddings
									  |
									  v
						 Cosine similarity ranking
									  |
						 Top matches and image grid
									  |
						 Optional FLUX room preview
```

## Repository Contents

```text
project.ipynb            Main notebook and Gradio application
requirements.txt         Python dependencies
room_*.jpg               Sample room inputs
embeddings/              Local embedding batches; generated files are ignored
.env.example             Environment variable template
```

The notebook currently contains saved outputs, including embedded image data. Review or clear notebook outputs before publishing if repository size, image licensing, or privacy is a concern.

## Getting Started

Create an environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `project.ipynb` in VS Code or Jupyter and run the setup and model-loading cells in order. On first use, the notebook downloads the CLIP model and the furniture dataset. A CUDA-enabled PyTorch installation is recommended for faster embedding and ranking.

## Optional Services

### Ollama

Install Ollama, start its local server, and make the model configured in the notebook available. The OpenAI-compatible endpoint is:

```text
http://localhost:11434/v1
```

### OpenRouter and FLUX previews

Preview generation is optional. Create a local `.env` file from `.env.example` and add your own key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Do not commit `.env`, API keys, model caches, generated previews, or embedding batches. The repository's `.gitignore` excludes these local artifacts.

For private demos, review the notebook's Gradio launch settings before running it. The current notebook uses `share=True`, which may expose the interface and uploaded room images through a temporary public share URL.

## Embedding Workflow

The notebook supports precomputed files such as `embeddings/batches_*.pt`. Keeping embeddings in batches makes the retrieval stage more manageable for limited-memory machines. If the batches are absent, generate them locally before running the ranking cells.

## Limitations and Next Steps

- Recommendation quality depends on the furniture dataset, CLIP similarity, and the room description produced by the vision model.
- The generated preview is a visual concept, not a dimensionally accurate interior-design plan.
- The notebook would benefit from pinned dependency versions, automated tests, a smaller clean demo notebook, and a dedicated application package.
