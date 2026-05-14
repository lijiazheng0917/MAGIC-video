# MAGIC-Video

[![arXiv](https://img.shields.io/badge/arXiv-2605.08271-b31b1b.svg)](https://arxiv.org/abs/2605.08271)
[![Dataset](https://img.shields.io/badge/🤗-Artifacts-yellow)](https://huggingface.co/datasets/jiazhengli7/magic-video-artifacts)

## 1. Introduction

**MAGIC-Video** is a training-free framework for ultra-long video reasoning (days to weeks of footage) built around a **M**ultimod**A**l memory **G**raph with **I**nterleaved narrative **C**hain. The **Multimodal Memory Graph (MMG)** unifies episodic captions, named entities, semantic triples, and visual clips into a single heterogeneous graph connected by six typed cross-modal and temporal edges, supporting cross-modal retrieval via a single Personalized PageRank pass. The **Narrative Memory Chain (NMC)** complements bottom-up graph aggregation with a top-down distillation that scans the whole video offline to surface per-entity *topic chains* (entity biographies) and multi-day *event chains* (recurring/multi-step activities) as coherent cross-time threads. At inference time, an agentic loop alternates between `search` and `answer` (capped at 5 search rounds), interleaving graph retrieval with narrative fact injection — covering both the modality and time dimensions of ultra-long video in a single retrieval pipeline. On three ultra-long video benchmarks, MAGIC-Video outperforms the strongest prior agentic systems by **+10.1** points on **EgoLifeQA**, **+7.4** points on **Ego-R1**, and **+5.9** points on **MM-Lifelong**.

<p align="center">
  <img src="figures/fig2.png" width="900" alt="Method overview">
</p>
<p align="center"><em>MAGIC-Video pipeline. <strong>Offline (left):</strong> preprocessing produces multi-granularity captions, named entities, semantic triples, and visual embeddings, from which we build the <strong>Multimodal Memory Graph</strong> (four node types connected by six typed edges) and the <strong>Narrative Memory Chain</strong> (topic chains + event chains). <strong>Online (right):</strong> for each question, an agentic loop seeds cross-modal Personalized PageRank over the graph, injects matching chains, and feeds the merged context to the reasoning backbone, which either refines its search or commits to an answer.</em></p>

---

## 2. Installation

```bash
git clone https://github.com/lijiazheng0917/MAGIC-video.git
cd MAGIC-video

# Environment
uv sync
source .venv/bin/activate

# Python path (required for every shell that runs eval/preprocess)
export PYTHONPATH=$PWD/src/EgoRAG:$PWD/src:${PYTHONPATH:-}
```

### API keys

All LLM calls go through OpenRouter. Export before running anything:

```bash
export OPENROUTER_API_KEY=<your-openrouter-key>
```

Default models used throughout the paper:

| Role | Model | Source |
|---|---|---|
| Retriever | `openai/gpt-oss-120b` | OpenRouter |
| Responder | `qwen/qwen3.5-flash-02-23` | OpenRouter |
| MM-Lifelong judge | `openai/gpt-5` (main) / `qwen/qwen3.5-flash-02-23` / `openai/gpt-5-mini` | OpenRouter |
| Text embedding | `Qwen/Qwen3-Embedding-4B` | Hugging Face (local, GPU) |

The text embedding model runs locally — weights are fetched from Hugging Face on first use. You can substitute any OpenRouter-compatible LLM or any Hugging Face sentence-embedding model via the CLI flags shown below.

### Optional: open-source VLM baselines

Reproducing the open-source VLM rows in the paper (Qwen3.5-9B, VideoLLaMA3, InternVideo2.5, LongVA, VideoChat-Flash) uses `vllm` and is **not covered by `uv sync`** — `vllm` has a different CUDA / PyTorch stack that conflicts with our main environment. Create a separate conda env for it:

```bash
conda create -n magic-video-baselines python=3.11 -y
conda activate magic-video-baselines
pip install vllm transformers accelerate
# plus the baseline-specific requirements, see baselines/eval/*.py
```

See [`baselines/README.md`](baselines/README.md) for per-model reproduction instructions. If you only care about reproducing **our** numbers (not the baselines), you can skip this.

### Compute requirements

Each command below is annotated with its resource needs:

| Tag | Meaning |
|---|---|
| `[API]` | OpenRouter calls only; runs on any CPU machine |
| `[GPU]` | Loads a local model (Whisper / VLM2Vec / Qwen-Embedding); needs an NVIDIA GPU |
| `[GPU + API]` | Both |
| `[Local]` | Pure file I/O / text processing; no GPU, no API |

All `[GPU]` steps in the paper were run on a single NVIDIA A100 (40 GB).

### Skipping preprocessing

All `[API]` steps produce JSON artifacts (captions, OpenIE results, semantic triples, topic and event chains). They cost OpenRouter credits and take several hours per subject / video, and because LLM outputs are non-deterministic, re-running them will give slightly different results from ours. To make reproduction both cheap and faithful, we release the exact artifacts used in the paper at [`jiazhengli7/magic-video-artifacts`](https://huggingface.co/datasets/jiazhengli7/magic-video-artifacts) (EgoLife `A1_JAKE` and MM-Lifelong). Once downloaded, you only need to run the `[GPU]` steps (visual embeddings + unified graph) and then go straight to evaluation.

In contrast, `[GPU]` steps are deterministic given the same inputs and model weights, so we do **not** ship them — you rebuild them locally with the commands below.

---

## 3. EgoLifeQA (500q MCQ)

Subject used in the paper: `A1_JAKE` — 7 days, **51.9 hours** of continuous first-person video. The 500 questions are split across five subtasks: EntityLog (EL), EventRecall (ER), HabitInsight (HI), RelationMap (RM), TaskMaster (TM).

### 3.1 Download `[Local]`

```bash
hf download lmms-lab/EgoLife --repo-type=dataset --local-dir data/EgoLife
```

### 3.2 Preprocess captions

```bash
# [API] Translate dense captions (CN → EN)
python data/EgoLife/utils/translate_densecap.py
# [Local] Align translated captions with transcripts
python data/EgoLife/utils/generate_sync.py
```

### 3.3 Extract memory features
```bash
# [API] Episodic memory: 30sec captions → multiscale → triples
python preprocess/egolife/episodic/generate_fine_caption.py \
    --sync-dir data/EgoLife/EgoLifeCap/Sync \
    --output data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json
python -m worldmm.memory.episodic.multiscale \
    --db_name A1_JAKE \
    --json_path data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json \
    --diary_dir .cache/events_diary \
    --save_path data/EgoLife/EgoLifeCap
python preprocess/egolife/episodic/extract_episodic_triples.py --subject A1_JAKE

# [API] Semantic memory triples
python preprocess/egolife/semantic/extract_semantic_triples.py --subject A1_JAKE
# [GPU + API] Consolidation (loads Qwen3-Embedding-4B for clustering + LLM for canonicalization)
python preprocess/egolife/semantic/consolidate_semantic_memory.py --subject A1_JAKE

# [GPU] Visual memory (VLM2Vec embeddings, single GPU)
CUDA_VISIBLE_DEVICES=0 python preprocess/egolife/visual/extract_visual_features.py --subject A1_JAKE --num_frames 16
# Multi-GPU alternative: bash preprocess/egolife/visual/extract_visual_features.sh --subject A1_JAKE --gpu 0,1,2,3 --num_frames 16
```

### 3.4 Build unified multimodal graph `[GPU]`

```bash
python preprocess/build_unified_graph.py \
    --dataset egolife --subject A1_JAKE \
    --embedding-model Qwen/Qwen3-Embedding-4B \
    --embedding-device cuda
```

### 3.5 Build Narrative Memory Chain `[API]`

The NMC has two extractors: per-entity **topic chains** and cross-time **event chains**.

```bash
python preprocess/egolife/extract_topic_chains.py \
    --subject A1_JAKE \
    --model openai/gpt-oss-120b \
    --output-dir output/metadata/topic_chains

python preprocess/egolife/extract_event_chains.py \
    --subject A1_JAKE \
    --model openai/gpt-oss-120b \
    --output-dir output/metadata/event_chains
```

### 3.6 Evaluate `[GPU + API]`

```bash
python eval/eval_egolife.py \
    --subject A1_JAKE \
    --retriever-model openai/gpt-oss-120b \
    --respond-model qwen/qwen3.5-flash-02-23 \
    --chain-mode facts \
    --topic-chain-facts-path output/metadata/topic_chains/A1_JAKE/topic_chains.json \
    --event-chain-path output/metadata/event_chains/A1_JAKE/step3_enriched_chains.json \
    --parallel 8
```

Chain hyperparameters use paper defaults (see `eval/eval_egolife.py` for the exact values). For the baseline (independent three-way retrieval), pass `--retrieval-backend independent --chain-mode ""`.

---

## 4. Ego-R1 (50q MCQ)

Ego-R1 reuses the EgoLife memory and graph — only the benchmark file differs.

### 4.1 Get the benchmark

Place the two split files at:
- `data/Ego-R1-Bench/manual-benchmark/A1_JAKE.json`
- `data/Ego-R1-Bench/gemini-benchmark/A1_JAKE.json`

See the Ego-R1 paper for download instructions.

### 4.2 Evaluate `[GPU + API]`

```bash
python eval/eval_egor1.py \
    --subject A1_JAKE \
    --retriever-model openai/gpt-oss-120b \
    --respond-model qwen/qwen3.5-flash-02-23 \
    --chain-mode facts \
    --topic-chain-facts-path output/metadata/topic_chains/A1_JAKE/topic_chains.json \
    --event-chain-path output/metadata/event_chains/A1_JAKE/step3_enriched_chains.json
```

Chain hyperparameters use paper defaults (see `eval/eval_egor1.py` for the exact values). Baseline: pass `--retrieval-backend independent --chain-mode ""`.

---

## 5. MM-Lifelong — Month subset (623q open-ended)

The paper uses the **Month** split — the longest temporal scale of MM-Lifelong (Day / Week / Month). It consists of **105.6 hours** of livestream video over **51 days**, and **623 open-ended questions** across 11 categories (Counting, Entity Recognition, Causal Reasoning, Temporal Reasoning, Event Recognition, Language Content Recall, Hallucination Detection, Attribute Recognition, Social Interaction, State Change, Event Tracking). Coverage of the 623 questions requires the **14 broadcast videos** below.

### 5.1 Download (14 videos — full coverage of the 623 val questions) `[Local]`

```bash
hf download CG-Bench/MM-Lifelong \
    month/4.mp4 month/11.mp4 month/12.mp4 month/13.mp4 \
    month/14.mp4 month/15.mp4 month/16.mp4 month/17.mp4 \
    month/18.mp4 month/19.mp4 month/20.mp4 month/21.mp4 \
    month/22.mp4 month/23.mp4 \
    --repo-type dataset --local-dir data/MM-Lifelong
```

For a quick-start subset you can download fewer videos (each covers a disjoint slice of questions):

| Subset | IDs | Questions covered |
|---|---|---|
| top3 | `14,19,18` | 222 / 623 (35.6%) |
| top5 | `14,19,18,21,17` | 336 / 623 (53.9%) |
| top7 | `14,19,18,21,17,23,22` | 442 / 623 (70.9%) |
| **all (paper)** | `4,11,12,13,14,15,16,17,18,19,20,21,22,23` | 623 / 623 (100%) |

### 5.2 Per-video preprocessing

For each broadcast `$vid`, run the full pipeline (ASR → VLM caption → merge → multi-scale → OpenIE → semantic → visual embeddings) in one command `[GPU + API]`:

```bash
python preprocess/mmlifelong/preprocess_video.py \
    --video-id $vid \
    --whisper-model large-v3-turbo \
    --llm-model openai/gpt-oss-120b
```

The script writes checkpoints after each stage, so reruns skip finished work.

<details>
<summary>Split into separate stages (optional, for HPC or to separate GPU vs API workloads)</summary>

```bash
# [GPU] Whisper ASR
python preprocess/mmlifelong/preprocess_video.py --video-id $vid --caption-mode asr \
    --skip-openie --skip-semantic --skip-visual

# [API] VLM caption via OpenRouter
python preprocess/mmlifelong/preprocess_video.py --video-id $vid --caption-mode vlm \
    --skip-whisper --skip-openie --skip-semantic --skip-visual

# [API] Merge ASR + VLM into multi-scale captions
python preprocess/mmlifelong/preprocess_video.py --video-id $vid --caption-mode merge \
    --skip-whisper --skip-vlm --skip-openie --skip-semantic --skip-visual

# [GPU + API] OpenIE + semantic extraction + consolidation
python preprocess/mmlifelong/preprocess_video.py --video-id $vid --caption-mode merge \
    --skip-whisper --skip-vlm --skip-captions --skip-visual

# [GPU] VLM2Vec visual embeddings
python preprocess/mmlifelong/preprocess_video.py --video-id $vid \
    --skip-whisper --skip-vlm --skip-captions --skip-openie --skip-semantic
```
</details>

### 5.3 Build unified graph (per video) `[GPU]`

```bash
python preprocess/build_unified_graph.py \
    --dataset mmlifelong \
    --video-id $vid \
    --embedding-model Qwen/Qwen3-Embedding-4B \
    --embedding-device cuda
```

### 5.4 Build Narrative Memory Chain (per video) `[API]`

```bash
python preprocess/mmlifelong/extract_topic_chains.py --video-id $vid --model openai/gpt-oss-120b
python preprocess/mmlifelong/extract_event_chains.py --video-id $vid --model openai/gpt-oss-120b
```

### 5.5 Evaluate `[GPU + API]`

```bash
python eval/eval_mmlifelong.py \
    --retriever-model openai/gpt-oss-120b \
    --respond-model qwen/qwen3.5-flash-02-23 \
    --chain-mode facts \
    --parallel 8
```

Chain hyperparameters use paper defaults (see `eval/eval_mmlifelong.py` for the exact values). Baseline: pass `--retrieval-backend independent --chain-mode ""`.

### 5.6 Re-judge with a different LLM `[API]`

```bash
python eval/rejudge.py \
    --input <results.json> \
    --output <results_judged_flash.json> \
    --judge-model qwen/qwen3.5-flash-02-23 \
    --parallel 8
```

---

## 6. Main Results

<p align="center">
  <img src="figures/result1.png" width="900" alt="Main benchmark results">
</p>
<p align="center"><em>Main results across EgoLifeQA, Ego-R1 and MM-Lifelong; rows correspond to the model categories reported in the paper.</em></p>

<p align="center">
  <img src="figures/result2.png" width="900" alt="MM-Lifelong per-question-type breakdown">
</p>
<p align="center"><em>MM-Lifelong per-question-type breakdown.</em></p>

See the paper for ablations, judge-comparison tables, and chain-injection breakdown.

---

## 7. Citation

Paper: [arXiv:2605.08271](https://arxiv.org/abs/2605.08271) · Artifacts: [jiazhengli7/magic-video-artifacts](https://huggingface.co/datasets/jiazhengli7/magic-video-artifacts)

```bibtex
@article{li2026magic,
  title  = {Bridging Modalities, Spanning Time: Structured Memory for Ultra-Long Agentic Video Reasoning},
  author = {Li, Jiazheng and Wu, Chi-Hao and Liu, Yunze and Ding, Kaize and Li, Jundong and Zhang, Chuxu},
  journal= {arXiv preprint arXiv:2605.08271},
  year   = {2026}
}
```

---

## 8. Acknowledgments

Built on [WorldMM](https://github.com/wgcyeo/WorldMM), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), and [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec). Datasets: [EgoLife](https://huggingface.co/datasets/lmms-lab/EgoLife) (LMMs-Lab), [Ego-R1](https://huggingface.co/datasets/Ego-R1/Ego-R1-Data), [MM-Lifelong](https://huggingface.co/datasets/CG-Bench/MM-Lifelong) (CG-Bench).


