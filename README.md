# Entropic RL: The Thermodynamic State (V2) 🌀

### *Solving the Homeostasis Problem in Coercive Systems*

This repository has transitioned from a deterministic simulation to an **Agent-Based Model** using **Entropy-Regularized Reinforcement Learning**. The "State" is now a neural network tasked with surviving a thermodynamic environment without triggering a systemic collapse.

---

## 💎 The Innovation: The Coercion Ratio ($R_c$)

While standard RL focuses on simple reward maximization, this project introduces the **Coercion Ratio ($R_c$)** as a governing metric for systemic health:

$$R_c(t) = \frac{E_{extracted}(t)}{\mathcal{H}_{citizens}(t) + \delta}$$

* **Low $R_c$ (< 1.0):** Homeostatic growth. The system is sustainable.
* **High $R_c$ (> 1.0):** Systemic Coercion. The state extracts more energy than the entropy of the citizenry can support, leading to the **"Heat Death"** observed in early models.

---

## 📊 V2 Results: The Stability Simplex

The current version implements a **Policy Gradient Agent** that learns to navigate the **Stability Simplex**. In the shaded green region below, the agent successfully balances its energy needs against the Coercion penalty.

![V2 Simulation Results](results.png)

*The red dashed line ($R_c$) represents the "stress" on the system. When $R_c$ spikes, Citizen Energy (blue) begins to crash.*

---

## 🏗️ Technical Implementation

### 1. The Homeostatic Reward Function
The agent is trained using a **Soft Objective** that internalizes the cost of coercion:
$$J(\pi) = \mathbb{E} [ \text{Energy} - \alpha R_c ]$$
By setting $\alpha = 0.5$, we force the "State" to treat citizen entropy as a finite, precious resource rather than an infinite battery.

### 2. Citizen Reaction Logic
In V2, citizens are no longer passive. The **Regrowth Function** is now tied to the Coercion Ratio. As $R_c$ increases, the citizenry's ability to regenerate energy diminishes exponentially, creating a feedback loop that punishes greed.

---

## 🚀 How to Run
```bash
python3 simulation.py
