#!/usr/bin/env python3
"""
Video recorder for evaluating and recording the best checkpoint.
Captures video of agent performance using gymnasium's video recording.
"""

import os
import argparse
import numpy as np
import torch
try:
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
except ImportError:
    import gym
    from gym.wrappers import Monitor as RecordVideo

from dinov2_transformerxl_ppo import PPOTrainer


def record_checkpoint_video(
    trainer: PPOTrainer,
    ckpt_path: str,
    env_id: str,
    output_dir: str = "videos",
    num_episodes: int = 5,
    deterministic: bool = True,
    max_steps_per_episode: int = 5000,
    video_prefix: str = "agent",
):
    """
    Record video of agent performance from a checkpoint.
    
    Args:
        trainer: PPOTrainer instance
        ckpt_path: Path to checkpoint file
        env_id: Gymnasium environment ID
        output_dir: Directory to save videos
        num_episodes: Number of episodes to record
        deterministic: Use deterministic actions
        max_steps_per_episode: Max steps per episode
        video_prefix: Prefix for video filenames
    """
    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)
    trainer.backbone.eval()
    trainer.projector.eval()
    trainer.policy.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment with video recording
    # Record every episode
    try:
        # Gymnasium style
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=output_dir,
            episode_trigger=lambda episode_id: True,  # Record all episodes
            name_prefix=video_prefix,
        )
    except TypeError:
        # Older gym style
        env = gym.make(env_id)
        env = RecordVideo(
            env,
            directory=output_dir,
            video_callable=lambda episode_id: True,
            name_prefix=video_prefix,
        )
    
    returns = []
    lengths = []
    
    # Initialize memory
    try:
        memory = trainer.memory[:, 0:1, :].to(trainer.device).clone()
    except Exception:
        memory = None
    
    print(f"Recording {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        # Reset environment
        obs_reset = env.reset()
        if isinstance(obs_reset, tuple) and len(obs_reset) >= 1:
            obs = obs_reset[0]
        else:
            obs = obs_reset
        
        done = False
        total_reward = 0.0
        steps = 0
        
        # Clone memory for this episode
        mem = memory.clone() if memory is not None else None
        
        print(f"Episode {ep + 1}/{num_episodes}...", end=" ", flush=True)
        
        while not done and steps < max_steps_per_episode:
            # Project observation
            proj = trainer.project_obs(np.expand_dims(obs, 0))  # [1, D]
            
            with torch.no_grad():
                # Forward pass through policy
                policy_out, value = trainer.policy.forward_sequence(
                    seq_proj=proj.unsqueeze(0), 
                    memory=mem
                )
                
                # Select action
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
                        std_np = std.squeeze(0).squeeze(0).cpu().numpy()
                        action = np.random.normal(mean, std_np)
            
            # Step environment
            step_ret = env.step(action)
            
            # Handle different return signatures
            if isinstance(step_ret, tuple) and len(step_ret) == 5:
                next_obs, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            elif isinstance(step_ret, tuple) and len(step_ret) == 4:
                next_obs, reward, done, info = step_ret
            else:
                raise RuntimeError(f"Unexpected env.step return: {step_ret}")
            
            total_reward += reward
            steps += 1
            
            # Update memory
            try:
                proj_token = proj.unsqueeze(0).cpu()
                if mem is not None:
                    mem = trainer.policy.update_memory(mem, proj_token.to(trainer.device))
            except Exception:
                pass
            
            obs = next_obs
        
        returns.append(total_reward)
        lengths.append(steps)
        print(f"Return: {total_reward:.2f}, Steps: {steps}")
    
    env.close()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("VIDEO RECORDING SUMMARY")
    print("="*50)
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Environment: {env_id}")
    print(f"Episodes recorded: {num_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"\nPerformance Statistics:")
    print(f"  Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Median Return: {np.median(returns):.2f}")
    print(f"  Min/Max Return: {np.min(returns):.2f} / {np.max(returns):.2f}")
    print(f"  Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"\nVideos saved to: {output_dir}/")
    print("="*50)
    
    return returns, lengths


def main():
    parser = argparse.ArgumentParser(
        description="Record video of agent performance from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ckpt_35840.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--video_prefix",
        type=str,
        default="agent",
        help="Prefix for video filenames",
    )
    
    args = parser.parse_args()
    
    # Create trainer instance
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Environment: {args.env}")
    
    trainer = PPOTrainer(
        env_id=args.env,
        num_envs=1,
        device=device,
        total_steps=1,  # Not training, just evaluating
        freeze_backbone=True,
    )
    
    # Record videos
    try:
        record_checkpoint_video(
            trainer=trainer,
            ckpt_path=args.checkpoint,
            env_id=args.env,
            output_dir=args.output_dir,
            num_episodes=args.episodes,
            deterministic=not args.stochastic,
            max_steps_per_episode=args.max_steps,
            video_prefix=args.video_prefix,
        )
    except Exception as e:
        print(f"\nError during recording: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
