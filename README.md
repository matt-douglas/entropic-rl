# Entropic RL: The Thermodynamic State (V2) 🌀

### *Solving the Homeostasis Problem in Coercive Systems*

This repository implements a **Thermodynamic Simulation** built on the principles of **Entropy-Regularized Reinforcement Learning (MaxEnt RL)**.

Unlike early versions which modeled a deterministic "Heat Death," Version 2 introduces an **Agentic State**—a neural network that must learn to navigate the stability constraints of its environment to avoid systemic liquidation.

## 💎 The Innovation: The Coercion Ratio ($R_c$)

Standard Reinforcement Learning focuses on reward maximization. This project introduces the **Coercion Ratio ($R_c$)** as the primary metric for systemic sustainability:

$$
R_c(t) = \frac{E_{extracted}(t)}{\mathcal{H}_{citizens}(t) + \delta}
$$

$R_c$ measures the "marginal cost of order." If the State extracts energy faster than the systemic entropy can buffer the cost, $R_c$ spikes, triggering a **Coercion Crisis**.

## 📊 V2 Results: The Stability Simplex

The plot below shows a trained **Policy Gradient Agent** discovering the **Homeostatic Zone**.

### Key Observations:

* **The Early Spike:** At the beginning of training, the agent is "greedy," causing $R_c$ (red dashed line) to spike and Citizen Energy (blue) to crash.

* **Learned Restraint:** As the **Soft Objective** penalizes high coercion, the agent learns to lower its extraction rate, allowing the system to settle into a "Sustainable Tyranny" where $R_c$ stays strictly below the **1.0 Stability Threshold**.

## 🧠 Technical Deep Dive

### 1. The Soft Objective (Reward)

The agent does not just maximize Energy ($E$). It optimizes a composite objective that internalizes the health of the host:

$$
J(\pi) = \mathbb{E} [ E_{state} - \alpha R_c ]
$$

Where $\alpha$ is the **Thermodynamic Temperature** controlling the trade-off between extraction and systemic stability.

### 2. Citizen Reaction Logic (Feedback)

In Version 2, citizens are no longer passive batteries. Their **Regrowth Rate** is inversely proportional to the current Coercion Ratio:

$$
\text{Regrowth} = \frac{\gamma}{1 + R_c}
$$

This creates an adversarial feedback loop: high coercion today poisons the resource base of tomorrow.

## 🚀 Execution & Usage

### Requirements

* Python 3.8+
* PyTorch
* Matplotlib

### Run the Agentic Loop

```bash
python3 simulation.py
