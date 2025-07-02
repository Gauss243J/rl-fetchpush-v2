import gym
import numpy as np
from stable_baselines3 import DDPG
import gym_robotics 

def run_episode(env, model, record_video=False, video_folder="./videos/", episode_id=0):
    if record_video:
        env_wrapped = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda e: True,
            name_prefix=f"episode_{episode_id}"
        )
    else:
        env_wrapped = env

    obs = env_wrapped.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_result = env_wrapped.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        if isinstance(obs, tuple):
            obs = obs[0]

        total_reward += reward
        steps += 1

    # Close only the wrapped environment
    env_wrapped.close()
    return total_reward, steps


env = gym.make("FetchPush-v2", render_mode="rgb_array")
model = DDPG.load("her_ddpg_fetchpush_v2", env=env)

num_episodes = 1000
batch_size = 100

all_rewards = []
all_steps = []

for batch_start in range(0, num_episodes, batch_size):
    batch_rewards = []
    batch_steps = []
    batch_episodes = []

    # Play batch_size episodes without video
    for ep in range(batch_start, batch_start + batch_size):
        reward, steps = run_episode(env, model, record_video=False)
        batch_rewards.append(reward)
        batch_steps.append(steps)
        print(f"Episode {ep+1}: Reward = {reward}, Steps = {steps}")

    # Find the episode with the best reward in this batch
    best_index = np.argmax(batch_rewards)
    best_reward = batch_rewards[best_index]
    best_ep = batch_start + best_index
    print(f"Best episode in batch {batch_start} to {batch_start + batch_size - 1}: episode {best_ep+1} with reward {best_reward}")

    # Replay the best episode WITH video recording
    print(f"Recording video for episode {best_ep+1} ...")
    run_episode(env, model, record_video=True, video_folder="./videos/", episode_id=best_ep)

    # Save the results
    all_rewards.extend(batch_rewards)
    all_steps.extend(batch_steps)

# Save the final stats
np.save("episode_rewards.npy", np.array(all_rewards))
np.save("episode_steps.npy", np.array(all_steps))

print("Process completed, videos of the best episodes recorded.")
