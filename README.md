
---

# RL-FetchPush2 – README
![image](https://github.com/user-attachments/assets/06e19edc-5ad4-4a8e-a954-2c2cb3ce5b0c) ![Untitled video - Made with Clipchamp](https://github.com/user-attachments/assets/fe3e00c2-a326-4212-82f1-b08a94535b34)


## Description

This project uses **Deep Deterministic Policy Gradient (DDPG)** combined with **Hindsight Experience Replay (HER)** to solve the FetchPush-v2 task from `gym_robotics` (OpenAI Gym).
The goal: train a robotic arm to push a box to a target location using reinforcement learning.

---

## Project Structure & File Roles

```
RL-FETCHPUSH2/
│
├── fetchvenv/                # Python virtual environment (project-specific libraries)
├── tb_fetchpush/             # TensorBoard logs (training progress visualization)
├── videos/                   # Saved videos of the best episodes (for demo)
│
├── episode_rewards.npy       # Rewards per episode (NumPy array, for performance analysis)
├── episode_steps.npy         # Steps per episode (NumPy array, for plotting/analysis)
│
├── fetchpush_train.py        # Main training script (DDPG+HER, saves the trained model)
├── her_ddpg_fetchpush_v2.zip # Trained model weights (saved checkpoint)
│
├── fetchpush_eval.py         # Automated evaluation: plays 1000 episodes, saves rewards/steps, records video of best episode
├── fetchpush_eval_realtime.py# Runs and displays 5 episodes in real time (with user instructions, MuJoCo window)
│
├── plot_training_progress.py # Plots training/evaluation curves (rewards and steps moving average)
│
├── test_fetchpush.py         # Simple script for quick training/testing (useful for debugging or rapid prototyping)
│
└── README.md                 # (this file)
```

---

## Script Details

* **fetchpush\_train.py**
  → Trains the DDPG + HER model on FetchPush-v2.
  → Uses Adam optimizer, batch size 100, HER replay buffer (n\_sampled\_goal=4).
  → Saves the trained model to `her_ddpg_fetchpush_v2.zip`.

* **fetchpush\_eval.py**
  → Evaluates the model for 1000 episodes.
  → Saves rewards and steps in `episode_rewards.npy` and `episode_steps.npy`.
  → Records a video of the best episode found.

* **fetchpush\_eval\_realtime.py**
  → Plays 5 episodes in real time with visualization using MuJoCo.
  → Provides instructions to the user (focus window, press H to hide help, etc.).

* **plot\_training\_progress.py**
  → Plots moving averages for rewards and steps using the `.npy` files.
  → Useful for analyzing training progress and showing results.

* **test\_fetchpush.py**
  → Simple script for quick testing or debugging (trains for 10,000 steps on FetchPush-v2).

---

## Data & Logs

* **episode\_rewards.npy / episode\_steps.npy**
  → NumPy files that save the total reward and step count for each evaluation episode.

* **tb\_fetchpush/**
  → Folder containing TensorBoard logs for advanced visualization (losses, rewards, hyperparameters, etc.).

* **videos/**
  → Directory where videos of the best episodes are saved (for demonstration or analysis).

---

## Environment

* **fetchvenv/**
  → Python virtual environment (recommended for dependency isolation: gym, stable-baselines3, gym-robotics, etc.).

---

## How to Run a Demo/Evaluation

1. **Train the model:**
   `python fetchpush_train.py`
2. **Evaluate and save best episode videos:**
   `python fetchpush_eval.py`
3. **Visualize performance curves:**
   `python plot_training_progress.py`
4. **Run a live demo:**
   `python fetchpush_eval_realtime.py`

---

## Author

Project by **Katembo Kaniki Joseph**
University of Klagenfurt, 2025

![Untitled video - Made with Clipchamp](https://github.com/user-attachments/assets/1e4d8bbb-7a7a-48c5-b485-2f497c43bcfb)


