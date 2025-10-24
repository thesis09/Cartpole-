[README.md](https://github.com/user-attachments/files/23130885/README.md)
Cartpole - DINO-v2 / ViT + Transformer-XL PPO

This repository contains a small research/experimental setup that trains a recurrent transformer (Transformer-XL style memory) policy on Gym environments using PPO. A Vision Transformer (ViT) backbone (or an identity backbone for vector observations) is used to extract features which are projected and fed to a recurrent policy.

Key artifacts
- `dinov2_transformerxl_ppo.py` — single-file trainer / model definitions and a small trainer (PPOTrainer).
- `eval_checkpoint.py` — evaluate all checkpoints in `checkpoints/` and write `eval_summary_<ts>.json`.
- `quick_eval_best.py` — convenience script that runs one checkpoint to inspect mean/std returns.
- `video_recorder.py` — records agent playback videos. Supports `--checkpoint BEST` to auto-discover the best checkpoint from `eval_summary_*.json` (or fallback to `quick_eval_best.py`).
- `checkpoints/` — saved model checkpoints (.pth files).

Goals
- Provide an experimental RL training loop that can work with image or vector observations.
- Make evaluation and playback robust and reproducible (seed support + deterministic options).

License
- See the `LICENSE` file in the repository root.
