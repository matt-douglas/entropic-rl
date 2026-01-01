# Phase 7: Swarm Orchestration (Thermodynamic Governance)

> "The single agent was the parable. The swarm is the empire."

This branch introduces the **Stochastic Multi-Agent Orchestrator**, moving beyond the linear constraints of the V2 single-agent model. It implements a **State-Dependent Thermodynamic System** that adapts its own entropy based on reward density.

## 📈 Optimization Post-Mortem: The 7 Phases of Convergence
The "Industrial Revolution" swarm underwent a rigorous thermodynamic tuning process to achieve sustainable throughput.

| Phase | Event | Entropy Strategy | Outcome |
| :--- | :--- | :--- | :--- |
| **1-2** | Initial Discovery | Slow Decay | Basic navigation established. |
| **3** | The Freeze | Premature Cooling | Reward: 0.00. Agents crystallized before discovery. |
| **3.1** | Thermal Runaway | Over-Correction | Reward: -868. System exploded into chaos. |
| **3.2** | The Quench | Brutal Cooling | Agents trapped in a Local Minimum (-1.0). |
| **4-5** | The Slingshot | High-Signal/High-Floor | Peak Reward (30.3) found but immediately overshot. |
| **6** | **The Velcro Lock** | **Brake-on-Reward** | **Peak Reward 68.07. High-precision exploitation.** |
| **7** | Homeostasis | Re-Heat Trigger | Sustainable adaptation to non-stationary rewards. |

## 💎 Key Breakthrough: The "Brake-on-Reward" Logic
The turning point was the transition from linear decay to **State-Dependent Thermodynamics**. By slashing entropy only upon high-reward detection (The "Kill-Switch"), we allowed the agents to remain fluid during the search but rigid during the harvest.

## Key Architecture
- **Gaussian RL Policies**: Agents operate with individual variance ($\sigma$) parameters.
- **Thermodynamic Governance**: A global temperature parameter ($T$) regulates the exploration/exploitation trade-off.
- **Peak-Relative Rescue**: The system tracks its historical maximum and triggers a "Re-Heat" event if utility drops below 75% of the peak.

## Usage
Run the simulation to generate the V3 metrics:
```bash
python swarm_sim.py
