# Phase 3: Swarm Orchestration (Thermodynamic Governance)

> "The single agent was the parable. The swarm is the empire."

This branch introduces the **Stochastic Multi-Agent Orchestrator**, moving beyond the linear constraints of the V2 single-agent model.

## Key Architecture
- **Gaussian RL Policies**: Agents operate with individual variance ($\sigma$) parameters that adapt based on local reward density.
- **Thermodynamic Governance**: A global temperature parameter ($T$) regulates the exploration/exploitation trade-off of the entire swarm.
- **Entropic Resilience**: The system operationalizes the "Maximum Entropy" principles defined in the core research to prevent premature convergence.

## Usage
Run the simulation to generate the V3 metrics:
```bash
python swarm_sim.py
