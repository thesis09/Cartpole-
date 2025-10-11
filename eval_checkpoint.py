#!/usr/bin/env python3


import os
import argparse
import json
import csv
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
try:
    import gymnasium as gym
except Exception:
    import gym
import matplotlib.pyplot as plt

# Import your trainer (adjust path if needed)
from dinov2_transformerxl_ppo import PPOTrainer

def safe_torch_load(path, map_location):
    # Try weights_only (newer PyTorch) else fallback
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # older torch doesn't have weights_only arg
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # fallback to normal load
        return torch.load(path, map_location=map_location)

def debug_print_actions_sample(all_actions, max_eps=3):
    print("=== Debug: sample action types/shapes for first episodes ===")
    for ei, ep_actions in enumerate(all_actions[:max_eps]):
        sample_info = []
        for ai, a in enumerate(ep_actions[:5]):
            sample_info.append((type(a).__name__, np.shape(a)))
        print(f"Episode {ei}: first_actions types/shapes: {sample_info}")

def evaluate_checkpoint(trainer: PPOTrainer, ckpt_path: str, env_id: str, num_episodes: int = 20, deterministic: bool = True, max_steps_per_episode: int = 1000):
    env = gym.make(env_id)
    # safe load with diagnostics
    try:
        ckpt = safe_torch_load(ckpt_path, map_location=trainer.device)
    except Exception as e:
        raise RuntimeError(f"Failed to torch.load checkpoint {ckpt_path}: {e}")

    # Try to load states safely; support multiple naming conventions
    try:
        # common keys
        if isinstance(ckpt, dict) and ('policy_state' in ckpt or 'policy' in ckpt or 'projector_state' in ckpt):
            # attempt to load sub-keys
            # load policy state if present
            if 'policy_state' in ckpt:
                missing, unexpected = trainer.policy.load_state_dict(ckpt['policy_state'], strict=False)
                print("policy load missing keys:", missing, "unexpected:", unexpected)
            elif 'policy' in ckpt:
                missing, unexpected = trainer.policy.load_state_dict(ckpt['policy'], strict=False)
                print("policy load missing keys:", missing, "unexpected:", unexpected)
            if 'projector_state' in ckpt:
                trainer.projector.load_state_dict(ckpt['projector_state'], strict=False)
            if 'backbone_state' in ckpt:
                trainer.backbone.load_state_dict(ckpt['backbone_state'], strict=False)
            if 'optimizer_state' in ckpt and hasattr(trainer, 'optimizer'):
                try:
                    trainer.optimizer.load_state_dict(ckpt['optimizer_state'])
                except Exception as ee:
                    print("Could not load optimizer state:", ee)
            # memory if available
            if 'memory' in ckpt:
                try:
                    trainer.memory = ckpt['memory'].to(trainer.device)
                except Exception:
                    trainer.memory = ckpt['memory']
        else:
            # If checkpoint is just a state_dict for a model, try to load into policy
            try:
                missing, unexpected = trainer.policy.load_state_dict(ckpt, strict=False)
                print("Loaded checkpoint as policy state_dict. missing:", missing, "unexpected:", unexpected)
            except Exception:
                # try loading into whole trainer
                try:
                    trainer_state = ckpt
                    # attempt keys if present
                    if 'policy_state_dict' in trainer_state:
                        trainer.policy.load_state_dict(trainer_state['policy_state_dict'], strict=False)
                    else:
                        # final fallback: ignore
                        pass
                except Exception:
                    pass
    except Exception as e:
        print("Warning: unexpected checkpoint structure. Proceeding, but you may see eval errors. Err:", e)

    trainer.backbone.eval(); trainer.projector.eval(); trainer.policy.eval()

    returns = []; lengths = []; all_actions = []; all_values = []
    try:
        memory = trainer.memory[:, 0:1, :].to(trainer.device).clone()
    except Exception:
        memory = None

    for ep in range(num_episodes):
        obs_reset = env.reset()
        # env.reset may return (obs, info) in Gymnasium; handle both forms
        if isinstance(obs_reset, tuple) and len(obs_reset) >= 1:
            obs = obs_reset[0]
        else:
            obs = obs_reset
        done = False; total = 0.0; length = 0
        episode_actions = []; episode_values = []
        mem = memory.clone() if memory is not None else None

        while (not done) and (length < max_steps_per_episode):
            proj = trainer.project_obs(np.expand_dims(obs, 0))  # [1, D]
            with torch.no_grad():
                try:
                    policy_out, value = trainer.policy.forward_sequence(seq_proj=proj.unsqueeze(0), memory=mem)
                except Exception as e:
                    # Dump debug and re-raise
                    print("Error running policy.forward_sequence:", e)
                    raise

                # sample action robustly
                if trainer.policy.is_discrete:
                    logits = policy_out.squeeze(0).squeeze(0)
                    if deterministic:
                        action = int(torch.argmax(logits).cpu().item())
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        action = int(torch.multinomial(probs, 1).cpu().item())
                else:
                    mean, std = policy_out
                    mean = mean.squeeze(0).squeeze(0).cpu().numpy()
                    if deterministic:
                        action = mean
                    else:
                        action = np.random.normal(mean, std.squeeze(0).squeeze(0).cpu().numpy())

                pred_value = float(value.squeeze(0).squeeze(0).cpu().item())

            step_ret = env.step(action)
            # support gymnasium (obs, reward, terminated, truncated, info)
            if isinstance(step_ret, tuple) and len(step_ret) == 5:
                next_obs, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            elif isinstance(step_ret, tuple) and len(step_ret) == 4:
                next_obs, reward, done, info = step_ret
            else:
                # unexpected form
                raise RuntimeError(f"Unexpected env.step return of length {len(step_ret) if isinstance(step_ret, tuple) else 'unknown'}")
            total += reward; length += 1
            episode_actions.append(action); episode_values.append(pred_value)

            # update memory safely
            try:
                proj_token = proj.unsqueeze(0).cpu()
                if mem is not None:
                    mem = trainer.policy.update_memory(mem, proj_token.to(trainer.device))
            except Exception:
                pass

            obs = next_obs

        returns.append(total); lengths.append(length)
        all_actions.append(episode_actions); all_values.append(episode_values)

    # debug print sample shapes
    debug_print_actions_sample(all_actions, max_eps=3)

    # safe aggregation
    flat_values = np.concatenate([np.array(v) for v in all_values if len(v) > 0]) if any(len(v)>0 for v in all_values) else np.array([])
    avg_pred_values = np.array([np.mean(v) if len(v) > 0 else 0.0 for v in all_values])
    episode_returns = np.array(returns)

    metrics = {
        "ckpt": ckpt_path,
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "value_pred_mean": float(np.mean(flat_values)) if flat_values.size>0 else None,
        "value_pred_std": float(np.std(flat_values)) if flat_values.size>0 else None,
        "episode_value_bias_mean": float(np.mean(avg_pred_values - episode_returns)),
        "episode_value_mse": float(np.mean((avg_pred_values - episode_returns) ** 2)),
    }

    # robust action stats
    if trainer.policy.is_discrete:
        flat_actions = np.concatenate([np.array(a, dtype=np.int64) for a in all_actions if len(a)>0]) if any(len(a)>0 for a in all_actions) else np.array([])
        if flat_actions.size>0:
            unique, counts = np.unique(flat_actions, return_counts=True)
            metrics["action_counts"] = {int(u): int(c) for u,c in zip(unique,counts)}
    else:
        action_arrays = []
        for a in all_actions:
            if len(a)==0:
                continue
            arr = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in a], axis=0)  # [steps,dim]
            action_arrays.append(arr)
        if len(action_arrays)>0:
            flat_actions = np.vstack(action_arrays)
            metrics["action_mean"] = list(np.mean(flat_actions, axis=0).tolist())
            metrics["action_std"] = list(np.std(flat_actions, axis=0).tolist())

    traces = {"returns": returns, "lengths": lengths, "actions": all_actions, "values": all_values}
    env.close()
    return metrics, traces

def find_checkpoints(folder):
    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.pth') or f.endswith('.pt')]
    files_sorted = sorted(files, key=os.path.getmtime)
    return files_sorted

def main(args):
    ckpts = find_checkpoints(args.checkpoints_dir)
    print(f"Found {len(ckpts)} checkpoints. Evaluating {args.episodes} eps each.")
    if len(ckpts)==0:
        return

    # create trainer with num_envs=1 for evaluation; freeze backbone to avoid accidental training
    trainer = PPOTrainer(env_id=args.env, num_envs=1, device=torch.device(args.device), total_steps=1, freeze_backbone=True)

    results = []
    for ck in ckpts:
        print("Evaluating:", ck)
        try:
            metrics, traces = evaluate_checkpoint(trainer, ck, env_id=args.env, num_episodes=args.episodes, deterministic=not args.stochastic)
            results.append((ck, metrics))
            print(f"-> {os.path.basename(ck)} mean_return={metrics['mean_return']:.2f} std={metrics['std_return']:.2f}")
        except Exception as e:
            print("Failed to evaluate", ck)
            traceback.print_exc()
            # save a small debug file for this ckpt
            with open(f"eval_error_{os.path.basename(ck)}.log", "w") as f:
                f.write(str(e) + "\n")
                f.write(traceback.format_exc())
            continue

    if len(results)==0:
        print("No successful evaluations.")
        return

    results_sorted = sorted([m for _,m in results], key=lambda x: x["mean_return"], reverse=True)
    best = results_sorted[0]
    try:
        with open("quick_eval_best.py", "r") as f:
            content = f.read()
        import re
        # Find any CKPT assignment line using regex
        new_content = re.sub(
            r'CKPT\s*=\s*"[^"]*"',
            f'CKPT = "{best["ckpt"]}"',
            content
        )
        with open("quick_eval_best.py", "w") as f:
            f.write(new_content)
        print(f"Updated quick_eval_best.py with best checkpoint: {best['ckpt']}")
    except Exception as e:
        print(f"Failed to update quick_eval_best.py: {e}")
    
    # save results
    ts = int(time.time())
    with open(f"eval_summary_{ts}.json", "w") as f:
        json.dump({"results": results_sorted}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    main(args)
