import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Citizen Logic (Sub-Agents)
class SubAgent:
    def __init__(self, id):
        self.id = id
        self.context_remaining = 100.0
        self.semantic_diversity = 1.0
        self.regrowth_rate = 0.8

    def process_task(self, task_complexity):
        response = " ".join(random.choices(["word" + str(i) for i in range(10)], k=int(task_complexity * 20)))
        self.context_remaining = max(0, self.context_remaining - task_complexity * 50)
        unique = len(set(response.split()))
        total = len(response.split())
        self.semantic_diversity = unique / total if total > 0 else 0.01
        return random.uniform(0.5, 1.0) if self.context_remaining > 0 else 0.0

    def recover(self, rc):
        effective = self.regrowth_rate / (1 + rc)
        self.context_remaining = min(100.0, self.context_remaining + effective)
        self.semantic_diversity = min(1.0, self.semantic_diversity + 0.02)

# 2. The Stochastic Orchestrator Policy (Phase 3)
class StochasticPolicy(nn.Module):
    def __init__(self, num_subs=3, state_dim=4):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, num_subs),
            nn.Softmax(dim=-1)
        )
        self.log_std = nn.Parameter(torch.zeros(num_subs))

    def forward(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        return mean, std

# 3. The Meta-Orchestrator
class MetaOrchestrator:
    def __init__(self, num_subs=3):
        self.num_subs = num_subs
        self.sub_agents = [SubAgent(i) for i in range(num_subs)]
        self.policy = StochasticPolicy(num_subs)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.alpha = 0.5  # Rc Penalty Temperature
        self.entropy_coeff = 0.05  # Bonus for exploration
        self.global_progress = 0.0
        self.history = []

    def get_state(self, task_complexity):
        avg_context = np.mean([s.context_remaining for s in self.sub_agents]) / 100
        avg_div = np.mean([s.semantic_diversity for s in self.sub_agents])
        return torch.tensor([task_complexity, avg_context, avg_div, self.global_progress / 100], dtype=torch.float32)

    def delegate_task(self, task_complexity):
        state = self.get_state(task_complexity)
        mean, std = self.policy(state)
        
        # Stochastic choice: Reparameterization/Sampling
        dist = torch.distributions.Normal(mean, std)
        action_samples = dist.sample()
        log_prob = dist.log_prob(action_samples).sum()
        entropy = dist.entropy().sum()
        
        sub_id = torch.argmax(action_samples).item()
        sub = self.sub_agents[sub_id]
        
        # Rc Calculation (The Owned Metric)
        anticipated_rc = task_complexity / (sub.semantic_diversity + 0.1)

        if anticipated_rc > 1.0 or sub.context_remaining < 5:
            # Reorg safety valve
            sub.context_remaining = 100.0
            sub.semantic_diversity = 1.0
            reward = -2.0  # Penalty for systemic failure
            rc = 5.0 # High stress
        else:
            quality = sub.process_task(task_complexity)
            rc = anticipated_rc
            reward = quality - (self.alpha * rc)

        # Update Policy (The Learning Swing)
        loss = -(reward + self.entropy_coeff * entropy) * log_prob
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_progress += reward
        for s in self.sub_agents:
            s.recover(rc)

        return reward, rc

def run_phase_3_sim(epochs=500):
    orchestrator = MetaOrchestrator()
    for t in range(epochs):
        task_complexity = random.uniform(0.3, 0.7)
        reward, rc = orchestrator.delegate_task(task_complexity)
        
        avg_context = np.mean([s.context_remaining for s in orchestrator.sub_agents])
        orchestrator.history.append([orchestrator.global_progress, avg_context, rc])
        
        if t % 50 == 0:
            print(f"Epoch {t} | Progress: {orchestrator.global_progress:.2f} | Rc: {rc:.2f}")

    # Plotting
    history = np.array(orchestrator.history)
    plt.figure(figsize=(10, 5))
    plt.plot(history[:, 0], label='Global Progress', color='indigo')
    plt.plot(history[:, 2], label='Coercion Ratio (Rc)', color='red', alpha=0.3, linestyle='--')
    plt.axhline(y=1.0, color='black', linestyle=':', label='Threshold')
    plt.legend()
    plt.title("Phase 3: Stochastic Swarm Orchestration")
    plt.savefig('swarm_v3_results.png')
    print("[SUCCESS] Phase 3 results saved to swarm_v3_results.png")

if __name__ == "__main__":
    run_phase_3_sim()
