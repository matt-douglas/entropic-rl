import numpy as np
import matplotlib.pyplot as plt

# --- Configuration: Phase 7.1 Peak-Relative Rescue ---
NUM_AGENTS = 50
EPOCHS = 200

# 1. Sustainable Floor
MIN_TEMP_FLOOR = 0.2
INITIAL_TEMP = 1.5

# 2. The Rescue Trigger (Peak-Relative)
# If reward drops below 75% of the historical max, we re-heat.
RETENTION_THRESHOLD = 0.75
REHEAT_TEMP = 2.0

# Velcro Settings
TARGET_REWARD = 10.0
BRAKE_FACTOR = 0.1
MIN_STEP = 0.01
BASE_STEP = 0.5
GOAL_STATE = 100.0
MAX_PENALTY = -1.0
SIGNAL_SCALAR = 10.0

class GaussianAgent:
    def __init__(self, id):
        self.id = id
        self.mu = 0.0
        self.sigma = 1.5
        self.state = 0.0
        self.last_reward = 0.0

    def act(self, temperature):
        noise = np.random.normal(0, self.sigma * temperature)
        action = np.random.normal(self.mu, 1.0) + noise
        return action

    def update(self, reward, learning_rate=0.05):
        self.mu += learning_rate * reward
        self.last_reward = reward
        
        if reward > 0:
            self.sigma = max(0.1, self.sigma * 0.95)
        else:
            self.sigma = min(3.0, self.sigma * 1.02)

class SwarmOrchestrator:
    def __init__(self):
        self.agents = [GaussianAgent(i) for i in range(NUM_AGENTS)]
        self.global_temperature = INITIAL_TEMP
        self.system_entropy = []
        self.mean_performance = []
        self.peak_reward = 0.0 # FIX: Memory of the Empire's Height

    def calculate_system_entropy(self):
        return np.mean([a.sigma for a in self.agents]) * self.global_temperature

    def run_epoch(self, epoch_idx):
        epoch_rewards = []
        
        for agent in self.agents:
            action = agent.act(self.global_temperature)
            
            # Velocity Control
            if agent.last_reward > 0:
                velocity = max(MIN_STEP, BASE_STEP / (1.0 + agent.last_reward))
            else:
                velocity = BASE_STEP
            
            agent.state += action * velocity
            
            # Reward Calculation
            dist = abs(GOAL_STATE - agent.state)
            base_reward = (100.0 - dist) / 10.0
            
            if base_reward > 0:
                final_reward = base_reward * SIGNAL_SCALAR
            else:
                final_reward = base_reward
            
            reward = max(MAX_PENALTY, final_reward)
            agent.update(reward)
            epoch_rewards.append(reward)

        avg_reward = np.mean(epoch_rewards)
        self.mean_performance.append(avg_reward)
        self.system_entropy.append(self.calculate_system_entropy())
        
        # --- PHASE 7.1: PEAK-RELATIVE LOGIC ---
        
        # 1. Update the Historical Peak
        self.peak_reward = max(self.peak_reward, avg_reward)
        
        status_tag = "TRACKING"

        # 2. The Rescue Trigger
        # Only trigger if we've actually established a peak (> TARGET)
        # AND we have fallen below 75% of that peak.
        if self.peak_reward > TARGET_REWARD and avg_reward < (self.peak_reward * RETENTION_THRESHOLD):
            self.global_temperature = REHEAT_TEMP
            status_tag = "RE-HEATING (RESCUE)"
            # Optional: Reset peak slightly so we don't get stuck in a loop
            self.peak_reward = avg_reward
        
        # 3. The Velcro Brake
        elif avg_reward > TARGET_REWARD:
            self.global_temperature *= BRAKE_FACTOR
            status_tag = "LOCKED (HARVESTING)"
        
        # 4. Standard Cooling
        else:
            self.global_temperature *= 0.98

        self.global_temperature = max(MIN_TEMP_FLOOR, self.global_temperature)
        
        return avg_reward, status_tag

def run_simulation():
    print(f"Initializing Phase 7.1 (Peak-Relative Rescue)...")
    print(f"Rescue Threshold: < {RETENTION_THRESHOLD*100}% of Peak")
    orchestrator = SwarmOrchestrator()
    
    for e in range(EPOCHS):
        avg_rew, status = orchestrator.run_epoch(e)
        entropy = orchestrator.system_entropy[-1]
        
        if e % 20 == 0:
            print(f"Epoch {e}: Reward = {avg_rew:.4f} | Entropy = {entropy:.4f} | Status: {status}")

    print("\nSimulation Complete. Generating visual proof...")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(orchestrator.mean_performance, color='#00ff99', linewidth=2, label='Mean Utility')
    plt.axhline(y=TARGET_REWARD, color='gold', linestyle='--', label='Lock Threshold')
    plt.title('Swarm Throughput (Resilient)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(orchestrator.system_entropy, color='#ff0055', linewidth=2, label='Entropy')
    plt.title('Thermodynamic Homeostasis', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('swarm_v3_results.png')
    print("Results saved to swarm_v3_results.png")

if __name__ == "__main__":
    run_simulation()
