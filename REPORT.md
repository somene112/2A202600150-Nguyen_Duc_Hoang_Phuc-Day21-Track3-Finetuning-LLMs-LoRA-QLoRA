# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Đức Hoàng Phúc — 2A202600150  
**Ngày nộp**: 2026-05-07  
**Submission option**: C / GitHub evidence

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples
- **Split**: 180 train / 20 eval
- **max_seq_length**: 1024
- **GPU**: Tesla T4, ~15.6 GB VRAM
- **Frameworks**: Unsloth, TRL, PEFT, Transformers, bitsandbytes
- **Estimated training cost**: ~$0.07, total training time ~12.0 minutes

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 3.83 | 8.70 | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.19 | 6.62 | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 3.97 | 9.48 | 1.4768 | 4.3790 |

## 3. Loss Curve Analysis

The training loss curve is included in `results/loss_curve.png`. Because T4 mode disabled evaluation during training to save VRAM, the curve mainly shows training loss. The run completed successfully without runtime errors. There is not enough mid-training eval data to strongly conclude overfitting, but the final eval losses across ranks suggest that larger rank improved perplexity on this small experiment.

## 4. Qualitative Comparison

The qualitative before/after generations are saved in `results/qualitative_comparison.csv`. Five prompts were tested, including machine learning explanation, Fibonacci code, UI/UX principles, LoRA vs QLoRA, and prompt engineering/RAG/fine-tuning comparison.

## 5. Conclusion về Rank Trade-off

In this experiment, rank 64 achieved the best eval loss and perplexity, but it also used the largest number of trainable parameters. Rank 8 was the lightest adapter, but its perplexity was worse than ranks 16 and 64. Rank 16 looks like the best balanced choice because it improved over rank 8 while keeping the adapter size much smaller than rank 64. For a small classroom experiment on Tesla T4, rank 16 is a reasonable default because it gives good quality with manageable VRAM and training cost. If the goal is only best validation perplexity and storage is not a concern, rank 64 is better. If the goal is production efficiency, rank 16 is safer.

## 6. What I Learned

- LoRA rank directly affects the number of trainable parameters and can change the quality/cost trade-off.
- QLoRA makes it possible to fine-tune a multi-billion parameter model on a free Colab T4 GPU.
- Evaluation should combine both numeric metrics such as perplexity and qualitative before/after examples.
