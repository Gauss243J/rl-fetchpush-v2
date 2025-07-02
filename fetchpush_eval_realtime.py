import gym
import gym_robotics
from stable_baselines3 import DDPG
import time

env = gym.make("FetchPush-v2", render_mode="human")
model = DDPG.load("her_ddpg_fetchpush_v2", env=env)

num_episodes = 5

print("="*60)
print("INSTRUCTIONS:")
print(" 1. When the MuJoCo window appears, CLICK inside it to focus.")
print(" 2. Press the 'H' key to hide the overlay menu/help text.")
print("    (This is required—MuJoCo does not support hiding it programmatically.)")
print(" 3. The agent will play 5 episodes in real time, with 3 seconds pause between episodes.")
print("="*60)

for ep in range(num_episodes):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False
    ep_reward = 0
    ep_steps = 0

    print(f"Starting Episode {ep+1}...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        if isinstance(obs, tuple):
            obs = obs[0]
        ep_reward += reward
        ep_steps += 1
        time.sleep(0.02)
    print(f"Episode {ep+1} finished. Reward: {ep_reward:.2f}, Steps: {ep_steps}")
    if ep < num_episodes - 1:
        print("Pausing for 5 seconds before the next episode...")
        time.sleep(3)

input("All episodes done! Press Enter to close the window and exit.")
env.close()
