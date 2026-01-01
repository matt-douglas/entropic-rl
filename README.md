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

---

## 🧠 Technical Deep Dive
<details>
<summary><b>Click to expand: The Math of Entropic RL</b></summary>

### 1. The Reparameterization Trick
In standard RL, the sampling operation $a \sim \pi(s)$ is non-differentiable. To allow backpropagation through the policy, we use the **Reparameterization Trick**. We express the action as a deterministic function of the state and independent noise $\epsilon$:

$$a = f_\theta(s, \epsilon) = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)$$

By shifting randomness to $\epsilon$, the gradient can flow directly from the Critic’s Q-value through the action to the Actor’s parameters $\theta$.

### 2. The Soft Bellman Equation
Unlike standard Q-Learning, Entropic RL optimizes for the **Soft Value Function**, which includes expected future entropy. The Soft Bellman backup operator $\mathcal{T}^\pi$ is:

$$\mathcal{T}^\pi Q(s, a) \approx r(s, a) + \gamma \mathbb{E}_{s' \sim P} [V(s')]$$

Where the Soft Value function $V(s')$ is defined as:
$$V(s') = \mathbb{E}_{a' \sim \pi} [Q(s', a') - \alpha \log \pi(a'|s')]$$

### 3. Numerical Stability & Squashing
To keep actions within a physical range (e.g., $[-1, 1]$), we apply a $tanh$ squashing function. This requires a **Jacobian Correction** to the log-probability to account for the change in density:

$$\log \pi(a|s) = \log \mu(u|s) - \sum_{i=1}^{D} \log(1 - \tanh^2(u_i))$$

This ensures that our entropy calculations remain accurate even after the action space is transformed.

</details>

---

## 📊 Results
[Insert a GIF or a plot showing the agent exploring different paths here!]

## 🏗️ Quick Start
```bash
git clone [https://github.com/matt-douglas/entropic-rl](https://github.com/matt-douglas/entropic-rl)
cd entropic-rl
pip install -r requirements.txt
python train.py --env "LunarLanderContinuous-v2"
