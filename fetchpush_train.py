import gym
import gym_robotics
from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer

env = gym.make("FetchPush-v2")

replay_buffer_kwargs = dict(
    n_sampled_goal=4,
    goal_selection_strategy="future"
)

model = DDPG(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=replay_buffer_kwargs,
    verbose=1,
    tensorboard_log="./tb_fetchpush/"
)

TIMESTEPS = 1_000_000
model.learn(total_timesteps=TIMESTEPS)
model.save("her_ddpg_fetchpush_v2")
print("Training completed and model saved as her_ddpg_fetchpush_v2.zip")
