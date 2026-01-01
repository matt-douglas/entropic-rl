# Entropic Reinforcement Learning

A clean, modular implementation of Entropy-Regularized Reinforcement Learning. This repository explores the "Control as Inference" framework, prioritizing robust exploration and multi-modal policy learning.

## 🚀 Why Entropic RL?
Standard RL often converges to a single deterministic path. Entropic RL (based on Maximum Entropy RL) encourages the agent to explore all possible ways to solve a task by maximizing:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

**Key Benefits:**
* **Superior Exploration:** Prevents early convergence to sub-optimal local minima.
* **Policy Robustness:** Learns a distribution of solutions, making it resilient to environment noise.
* **Transfer Learning:** The learned features are more generalizable than greedy policies.

## 🛠️ Features
* **Soft Actor-Critic (SAC) Core:** Implementation of the off-policy maximum entropy actor-critic algorithm.
* **Automated Temperature Tuning:** Dynamic adjustment of the entropy coefficient $\alpha$.
* **Modular Design:** Easy to swap environments (Gym/Farama) and neural architectures.

## 📊 Results
[Insert a GIF or a plot showing the agent exploring different paths here!]

## 🏗️ Quick Start
```bash
git clone [https://github.com/matt-douglas/entropic-rl](https://github.com/matt-douglas/entropic-rl)
cd entropic-rl
pip install -r requirements.txt
python train.py --env "LunarLanderContinuous-v2"
