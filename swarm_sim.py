import numpy as np
import matplotlib.pyplot as plt

# --- Configuration: Phase 7 Sustainable Harvest ---
NUM_AGENTS = 50
EPOCHS = 200

# 1. Sustainable Floor (Raised from 0.01 to 0.2)
# We never let the system freeze completely; it must remain "liquid" enough to adapt.
MIN_TEMP_FLOOR = 0.2
INITIAL_TEMP = 1.5

# 2. The Hunger Trigger (Re-Heat Logic)
REHEAT_DROP_THRESHOLD = 0.15  # 15% drop triggers re-heat
REHEAT_TEMP = 2.0             # The temperature injection

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
        self.prev_epoch_reward = 0.0 # Memory for the "Hunger" check

    def calculate_system_entropy(self):
        return np.mean([a.sigma for a in self.agents]) * self.global_temperature

    def run_epoch(self, epoch_idx):
        epoch_rewards = []
        
        for agent in self.agents:
            # Action
            action = agent.act(self.global_temperature)
            
            # Velocity Control (Inverse to Reward)
            if agent.last_reward > 0:
                velocity = max(MIN_STEP, BASE_STEP / (1.0 + agent.last_reward))
            else:
                velocity = BASE_STEP
            
            agent.state += action * velocity
            
            # Reward Calculation (Simulating Resource Depletion check)
            # In a real dynamic env, the GOAL_STATE might move.
            # Here, we simulate the "need" to re-verify by tracking drops.
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
        
        # --- PHASE 7: SUSTAINABLE LOGIC ---
        
        # 1. Calculate Drop
        # Avoid division by zero on first run
        if self.prev_epoch_reward > 0:
            drop_ratio = (self.prev_epoch_reward - avg_reward) / self.prev_epoch_reward
        else:
            drop_ratio = 0.0
            
        status_tag = "TRACKING"

        # 2. The Hunger Trigger
        if drop_ratio > REHEAT_DROP_THRESHOLD:
            # Resource Exhaustion Detected! Re-Heat immediately.
            self.global_temperature = REHEAT_TEMP
            status_tag = "RE-HEATING (HUNGER)"
        
        # 3. The Velcro Brake (Only applies if NOT re-heating)
        elif avg_reward > TARGET_REWARD:
            self.global_temperature *= BRAKE_FACTOR
            status_tag = "LOCKED (HARVESTING)"
        
        # 4. Standard Cooling
        else:
            self.global_temperature *= 0.98

        # 5. Persistent Floor
        self.global_temperature = max(MIN_TEMP_FLOOR, self.global_temperature)
        
        self.prev_epoch_reward = avg_reward
        
        return avg_reward, status_tag

def run_simulation():
    print(f"Initializing Phase 7 (Sustainable Harvest)...")
    print(f"Re-Heat Threshold: {REHEAT_DROP_THRESHOLD*100}% Drop")
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
    plt.title('Swarm Throughput (Adaptive)', fontsize=14)
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
