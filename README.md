# Entropic RL: The Thermodynamic State 🌀

A simulation exploring the intersection of **Entropy-Regularized Reinforcement Learning** and **Thermodynamic Systems**. This project models a "Digital State" where policy stability is governed by the balance of energy, entropy, and systemic decay.

---

## 📊 Simulation Results: The Rise and Fall
The current simulation models a "thermodynamic government." As the State Energy increases, it often consumes the energy of its citizens, leading to a point of zero-citizen entropy and eventual systemic collapse.


> **Key Observation:** Year 500-2500 shows the transition from a high-energy state to a "heat death" scenario where the system can no longer sustain its complexity.

---

## 🧠 Technical Deep Dive
<details>
<summary><b>Click to expand: The Math of Entropic RL</b></summary>

### 1. The Reparameterization Trick
To allow backpropagation through the random sampling of actions, we use the **Reparameterization Trick**:

$$a = f_\theta(s, \epsilon) = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)$$

### 2. The Soft Bellman Equation
This project utilizes the **Soft Value Function**, which rewards the agent for both success and maintaining high entropy (diversity of states):

$$V(s') = \mathbb{E}_{a' \sim \pi} [Q(s', a') - \alpha \log \pi(a'|s')]$$

Where $\alpha$ is the "temperature" parameter controlling the trade-off between reward and exploration.

</details>

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.8+ installed. 
```bash
pip install -r requirements.txt
