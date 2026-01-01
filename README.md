# Entropic RL: The Thermodynamic State 🌀

### *Where Control Theory meets Social Physics*

This repository implements a **Thermodynamic Simulation** built on the principles of **Entropy-Regularized Reinforcement Learning (MaxEnt RL)**. It models a "Digital State" as a closed system where the agent (the State) must balance its own energy accumulation against the entropic vitality of its citizens.

---

## 📊 The Simulation: Thermodynamic Government

The simulation (found in `simulation.py`) tracks the evolution of a system over 2,500 iterations. It explores the **Exploitation vs. Exploration** trade-off through a thermodynamic lens.

### **The "Heat Death" Observation**
In the current **Recovery Patch**, we observe a systemic collapse:
* **Years 0–500:** The State successfully accumulates energy (Exploitation), but at the cost of depleting **Citizen Energy**.
* **Years 500–2500:** As Citizen Energy hits **0.0**, the system loses its "temperature." Without the noise/entropy provided by the citizens, the State energy begins a slow, entropic decay toward systemic "Heat Death."

![Simulation Results](results.png?v=10)

---

## 🧠 Technical Deep Dive: RL as Inference

This project is based on the **Control as Inference** framework. Instead of simply maximizing a reward sum, the agent maximizes a **Soft Objective** that values diversity of behavior.



### 1. The Maximum Entropy Objective
Standard RL optimizes for $\sum R_t$. This project utilizes the **MaxEnt** objective, which adds an information-theoretic penalty to the agent's behavior to ensure continuous exploration:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

* **Reward ($r$):** State stability and energy accumulation.
* **Entropy ($\mathcal{H}$):** The diversity and freedom of citizen states.
* **Temperature ($\alpha$):** Determines if the system prioritizes "order" (reward) or "chaos" (entropy).

### 2. The Reparameterization Trick (Path Gradients)
To train neural networks in an entropic environment, we must differentiate through the sampling process. We use the **Reparameterization Trick**, treating the action $a$ as a deterministic transformation of independent Gaussian noise $\epsilon$:

$$a = f_\theta(s, \epsilon) = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)$$



This allows the gradient to flow through the stochastic policy, enabling the "Government" to optimize its policy via Gradient Descent.

### 3. Soft Bellman Equation
The system learns using a **Soft Value Function** $V(s)$, which represents the "Free Energy" of the current state:

$$V(s') = \mathbb{E}_{a' \sim \pi} [Q(s', a') - \alpha \log \pi(a'|s')]$$

By subtracting the log-probability, we penalize the agent for becoming too deterministic (low entropy), effectively forcing it to discover more robust strategies.

---

## 🚀 Execution & Usage

### **Requirements**
* Python 3.8+
* PyTorch (Neural Engine)
* Matplotlib (Visualization)

```bash
# Clone and Install
git clone [https://github.com/matt-douglas/entropic-rl](https://github.com/matt-douglas/entropic-rl)
cd entropic-rl
python3 -m pip install -r requirements.txt

# Run the Simulation
python3 simulation.py
