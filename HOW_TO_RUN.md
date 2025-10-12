HOW TO RUN (for GitHub / local dev)

This file explains how to set up and run the code in this repo. The instructions target a Unix-like shell (Linux or macOS).
Windows users: use PowerShell or WSL and adapt paths/activation commands.

1) Create & activate a Python virtual environment (recommended)
---------------------------------------------------------------
# create environment
python -m venv .venv

# activate (Linux / macOS)
source .venv/bin/activate

# activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

2) Install dependencies
-----------------------
pip install --upgrade pip
pip install -r requirements.txt

# If you need CUDA-enabled PyTorch, install the proper wheel from pytorch.org before or instead of the line above.
# Example for CUDA 11.8 (Linux): 
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

3) Quick sanity check (small training run)
------------------------------------------
# Run a short demo-training using CartPole (vector observations)
python dinov2_transformerxl_ppo.py

# The default script settings produce a lightweight run; adjust constants in the file (NUM_ENVS, TOTAL_STEPS) for longer runs.

4) Evaluate checkpoints
-----------------------
# Evaluate all *.pth / *.pt files in checkpoints/ (creates JSON summary and prints results)
python eval_checkpoint.py --checkpoints_dir checkpoints --env CartPole-v1 --episodes 20

# Use --stochastic if you want the policy to sample actions rather than act deterministically.
# Example:
# python eval_checkpoint.py --checkpoints_dir checkpoints --env CartPole-v1 --episodes 20 --stochastic

5) Quick deterministic playback
-------------------------------
# Edit CKPT path in quick_eval_best.py to point at a saved checkpoint and run:
python quick_eval_best.py

6) Notes for image-based training
---------------------------------
- The default ENV in the scripts may be CartPole-v1 (non-image observation). If you want to train a ViT (vision transformer),
  use an environment that returns images (e.g., Atari wrappers, environments with render_mode='rgb_array', or any custom env that emits HxWx3 images).
- The backbone loader supports 'vit_b_16' pretrained from torchvision. DINO-v2 support is marked as TODO and requires loading a DINO checkpoint
  and mapping weights to the vit model.

7) Checkpointing & resuming
---------------------------
- Checkpoints are saved in the `checkpoints/` folder by default.
- Checkpoint load/save handles several common formats but be careful if you change model shapes/hyperparams.

8) Reproducibility & seeds
--------------------------
- The code seeds each sub-env using base_seed + index. To fully reproduce results you should set
  seeds for: numpy, torch, python's random, and ensure deterministic cudnn flags if needed.

9) Troubleshooting
------------------
- "ModuleNotFoundError" : ensure you activated the virtual environment and installed requirements.
- GPU memory errors: reduce `transformer_d_model`, `transformer_layers`, `minibatch_size`, or use `freeze_backbone=True`.
- For Windows + GPU: install the correct torch wheel; mismatched CUDA versions cause runtime errors.

10) Contact / next steps
------------------------
If you want:
- a script that wraps a visual environment around CartPole (returns frames) — I can add it,
- DINO-v2 checkpoint loader implemented — provide the checkpoint or tell me the format and I'll implement the loader.

