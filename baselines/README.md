# VLM Baselines

Reproduction instructions for the open-source and API-based VLM baselines reported in the paper (MM-Lifelong / EgoLife / Ego-R1).

All baseline numbers sit under:
```
output/eval/baselines/<benchmark>_<model>_<config>/results.json
```

---

## 1. Environment

Most open-source VLMs require a **separate conda environment** from the main project — their CUDA / PyTorch stack conflicts with ours. Pick one of the two envs below depending on the model.

### 1a. vLLM env (Qwen3.5-9B)
Used by `baselines/eval/eval_vlm_egolife.py` and `baselines/eval/eval_vlm_mmlifelong.py` when driven by a local vLLM server.

```bash
conda create -n magic-video-baselines python=3.11 -y
conda activate magic-video-baselines
pip install vllm transformers accelerate
```

Launch the vLLM server separately (port 8000) and point `--api-base http://localhost:8000/v1` to it. The `eval_vlm_*.py` scripts speak the OpenAI-compatible protocol.

### 1b. Video-LLM env (VideoLLaMA3 / InternVideo2.5 / LongVA / VideoChat-Flash)
These models need an older torch/transformers combo.

```bash
conda create -n magic-video-videollama python=3.10 -y
conda activate magic-video-videollama
pip install torch==2.4.0+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.46.3" accelerate flash-attn==2.7.3 decord pysrt opencv-python
```

### 1c. API baselines (GPT-5 Mini, Gemini 3 Flash)
Just need the main project env + `OPENROUTER_API_KEY`. No conda needed.

---

## 2. Pre-extract frames (once per benchmark)

The video-LLM baselines load from a frame cache to keep evaluation deterministic and fast.

```bash
# EgoLife (500q) — extracts 64 frames per 30-sec clip
python baselines/eval/preextract_egolife_videos.py \
    --num-frames 64 \
    --output-dir output/metadata/baselines/egolife_videos_64f

# Ego-R1 (50q) — reuses EgoLife videos, different QA file
python baselines/eval/preextract_egolife_videos.py \
    --num-frames 64 \
    --qa-path data/Ego-R1-Bench/combined_A1_JAKE.json \
    --output-dir output/metadata/baselines/egor1_videos_64f

# MM-Lifelong (623q) — 64 frames per video, writes manifest.json
python baselines/eval/preextract_mml_frames.py \
    --num-frames 64 \
    --output-dir output/metadata/baselines/mml_frames_64f
```

---

## 3. Run a baseline

### 3.1 Qwen3.5-9B (V+T) via vLLM

Start the vLLM server in one shell:
```bash
conda activate magic-video-baselines
vllm serve Qwen/Qwen2.5-VL-9B-Instruct --port 8000 --max-model-len 100000 --gpu-memory-utilization 0.95
```

Then in another shell (main project env):
```bash
# MM-Lifelong
python baselines/eval/eval_vlm_mmlifelong.py \
    --model qwen/qwen2.5-vl-9b-instruct \
    --mode frames+caption --num-frames 64 \
    --frames-dir output/metadata/baselines/mml_frames_64f \
    --api-base http://localhost:8000/v1 \
    --output-dir output/eval/baselines/mmlifelong_qwen35_9b_64f \
    --parallel 1

# EgoLife
python baselines/eval/eval_vlm_egolife.py \
    --model qwen/qwen2.5-vl-9b-instruct \
    --mode frames --num-frames 64 \
    --frame-cache output/metadata/baselines/egolife_videos_64f/frame_cache \
    --api-base http://localhost:8000/v1 \
    --output-dir output/eval/baselines/egolife_qwen35_9b_64f
```

### 3.2 VideoLLaMA3-7B / InternVideo2.5-8B / LongVA-7B / VideoChat-Flash-7B

Run from the Video-LLM env (no vLLM server):
```bash
conda activate magic-video-videollama

# VideoLLaMA3 on MM-Lifelong
python baselines/eval/eval_videollama3.py \
    --model DAMO-NLP-SG/VideoLLaMA3-7B \
    --num-frames 128 \
    --benchmark mmlifelong \
    --mml-frames-dir output/metadata/baselines/mml_frames_64f \
    --output-dir output/eval/baselines/mmlifelong_videollama3_128f

# InternVideo2.5 on MM-Lifelong
python baselines/eval/eval_internvideo.py \
    --model OpenGVLab/InternVideo2_5_Chat_8B \
    --num-frames 512 \
    --benchmark mmlifelong \
    --mml-frames-dir output/metadata/baselines/mml_frames_64f \
    --output-dir output/eval/baselines/mmlifelong_internvideo_512f

# LongVA
python baselines/eval/eval_longva.py \
    --model lmms-lab/LongVA-7B \
    --num-frames 128 \
    --benchmark mmlifelong \
    --mml-frames-dir output/metadata/baselines/mml_frames_64f \
    --output-dir output/eval/baselines/mmlifelong_longva_128f

# VideoChat-Flash
python baselines/eval/eval_videochat_flash.py \
    --model OpenGVLab/VideoChat-Flash-Qwen2_5-7B_res448 \
    --num-frames 1024 \
    --benchmark mmlifelong \
    --mml-frames-dir output/metadata/baselines/mml_frames_64f \
    --output-dir output/eval/baselines/mmlifelong_vcflash_1024f
```

Swap `--benchmark` to `egolife` or `egor1` and point `--frame-cache output/metadata/baselines/egolife_videos_64f/frame_cache` (plus `--qa-path data/Ego-R1-Bench/combined_A1_JAKE.json` for Ego-R1) to evaluate on those benchmarks.

### 3.3 API baselines (GPT-5 Mini / Gemini 3 Flash)

Main project env + `OPENROUTER_API_KEY`:
```bash
# GPT-5 Mini on MM-Lifelong (reasoning-effort 'low' matches paper)
python baselines/eval/eval_vlm_mmlifelong.py \
    --model openai/gpt-5-mini \
    --reasoning-effort low \
    --mode frames+caption --num-frames 64 \
    --frames-dir output/metadata/baselines/mml_frames_64f \
    --judge-model openai/gpt-5 \
    --output-dir output/eval/baselines/mmlifelong_gpt5mini_fc_low \
    --parallel 4

# Gemini 3 Flash on MM-Lifelong
python baselines/eval/eval_vlm_mmlifelong.py \
    --model google/gemini-3-flash-preview \
    --reasoning-effort low \
    --mode frames+caption --num-frames 64 \
    --frames-dir output/metadata/baselines/mml_frames_64f \
    --judge-model openai/gpt-5 \
    --output-dir output/eval/baselines/mmlifelong_gemini3flash_fc_low \
    --parallel 4
```

---

## 4. Re-judge an existing baseline run

All MM-Lifelong baselines report both Flash and GPT-5 judge scores. If you already have `results.json` (inline judged), re-judge with a different LLM:

```bash
python eval/rejudge.py \
    --input output/eval/baselines/mmlifelong_<model>_<config>/results.json \
    --output output/eval/baselines/mmlifelong_<model>_<config>/results_judged_<newjudge>.json \
    --judge-model <openai/gpt-5 | qwen/qwen3.5-flash-02-23 | openai/gpt-5-mini> \
    --parallel 8
```

---

## 5. Frame / context sizes used in the paper

| Model | Benchmark | Frames | Mode |
|---|---|---|---|
| Qwen3.5-9B (V) | all | 64 | `frames` |
| Qwen3.5-9B (T) | all | 0 (captions only, 512 max) | `caption` |
| Qwen3.5-9B (V+T) | all | 64 + captions | `frames+caption` |
| Qwen3.5-Flash | MML | 64 + captions | `frames+caption` |
| GPT-5 Mini / Gemini 3 Flash | MML | 64 + captions | `frames+caption`, `--reasoning-effort low` |
| VideoLLaMA3-7B | MML | 128 | — |
| InternVideo2.5-8B | MML | 512 | — |
| LongVA-7B | MML | 128 | — |
| VideoChat-Flash-7B | MML | 1024 | — |

See Table 2 in the paper for per-question-type breakdowns.
