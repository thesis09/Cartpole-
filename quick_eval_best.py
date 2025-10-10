# quick_eval_best.py
from dinov2_transformerxl_ppo import PPOTrainer
import numpy as np, torch
try:
    import gymnasium as gym
except Exception:
    import gym

CKPT = "checkpoints/ckpt_35840.pth"
ENV = "CartPole-v1"   # replace with your env
EPISODES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = PPOTrainer(env_id=ENV, num_envs=1, device=device, total_steps=1, freeze_backbone=True)
trainer.load_checkpoint(CKPT)
trainer.backbone.eval(); trainer.projector.eval(); trainer.policy.eval()

returns = []
for ep in range(EPISODES):
    env = gym.make(ENV)
    obs_reset = env.reset()
    # handle gymnasium reset signature (obs, info)
    if isinstance(obs_reset, tuple) and len(obs_reset) >= 1:
        obs = obs_reset[0]
    else:
        obs = obs_reset
    done = False
    R = 0.0
    steps = 0
    memory = trainer.memory[:, 0:1, :].to(trainer.device).clone() if hasattr(trainer, "memory") else None
    while not done and steps < 5000:
        proj = trainer.project_obs(np.expand_dims(obs, 0))  # [1,D]
        with torch.no_grad():
            policy_out, _ = trainer.policy.forward_sequence(seq_proj=proj.unsqueeze(0), memory=memory)
            if trainer.policy.is_discrete:
                logits = policy_out.squeeze(0).squeeze(0)
                action = int(torch.argmax(logits).cpu().item())
            else:
                mean, _ = policy_out
                action = mean.squeeze(0).squeeze(0).cpu().numpy()
        # step once on the same env; handle gymnasium step signature
        step_ret = env.step(action)
        if isinstance(step_ret, tuple) and len(step_ret) == 5:
            next_obs, r, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        elif isinstance(step_ret, tuple) and len(step_ret) == 4:
            next_obs, r, done, info = step_ret
        else:
            raise RuntimeError(f"Unexpected env.step return: {step_ret}")
        R += r
        steps += 1
        # update memory
        try:
            proj_token = proj.unsqueeze(0).cpu()
            if memory is not None:
                memory = trainer.policy.update_memory(memory, proj_token.to(trainer.device))
        except Exception:
            pass
        obs = next_obs
    env.close()
    print(f"Episode {ep}: return={R:.2f}, steps={steps}")
    returns.append(R)

print("DET eval mean/std:", np.mean(returns), np.std(returns))
