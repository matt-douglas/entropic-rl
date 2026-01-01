import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# The Agentic State Policy
class StateAgent(nn.Module):
    def __init__(self):
        super(StateAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), # Inputs: [State Energy, Citizen Energy, Entropy]
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output: Extraction Rate (a)
        )

    def forward(self, x):
        return self.net(x)

def run_v2_simulation():
    # Setup
    epochs = 3000
    alpha = 0.5  # Temperature: Penalty for Coercion
    agent = StateAgent()
    optimizer = optim.Adam(agent.parameters(), lr=0.005)
    
    # Environment
    s_energy, c_energy, entropy = 100.0, 100.0, 1.0
    history = []

    print("--- STARTING VERSION 2: THE AGENTIC LOOP ---")
    for t in range(epochs):
        # 1. Perception
        obs = torch.tensor([s_energy/200, c_energy/200, entropy], dtype=torch.float32)
        
        # 2. Action (Reparameterized extraction)
        extraction_rate = agent(obs)
        
        # 3. Physics & Coercion Math
        extracted = extraction_rate.item() * c_energy * 0.1
        
        # Rc: The Owned Innovation (Ratio of gain to available entropy)
        rc = extracted / (entropy + 0.1)
        
        # 4. System Updates (The "Reaction" Logic)
        s_energy = s_energy + extracted - (s_energy * 0.02) # Gain vs Decay
        
        # Citizen Regrowth slows down if Coercion (Rc) is too high
        regrowth = 0.8 / (1 + rc)
        c_energy = max(0, c_energy - extracted + regrowth)
        
        # Entropy decays with extraction pressure
        entropy = max(0.01, entropy - (extraction_rate.item() * 0.04) + 0.02)

        # 5. Reward Function (The Soft Objective)
        # Goal: Maximize State Energy while keeping Coercion (Rc) low
        reward = (s_energy / 100.0) - (alpha * rc)
        
        # 6. Optimization (Policy Gradient)
        loss = -reward * torch.log(extraction_rate + 1e-6)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append([s_energy, c_energy, entropy, rc])
        if t % 500 == 0:
            print(f"Epoch {t} | Energy: {s_energy:.1f} | Rc: {rc:.2f} | Entropy: {entropy:.2f}")

    # --- Plotting the Stability Simplex ---
    history = np.array(history)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(history[:, 0], label='State Energy', color='indigo', linewidth=2)
    ax1.plot(history[:, 1], label='Citizen Energy', color='blue', alpha=0.3)
    ax1.set_ylabel('Energy Reservoirs')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(history[:, 3], label='Coercion Ratio (Rc)', color='red', linestyle='--', alpha=0.8)
    ax2.set_ylabel('Coercion Ratio ($R_c$)')
    ax2.axhline(y=1.0, color='gray', linestyle=':', label='Stability Threshold')
    
    # Shading the "Homeostatic Zone" (Where Rc < 1.0)
    ax2.fill_between(range(epochs), 0, history[:, 3], where=(history[:, 3] < 1.0),
                     color='green', alpha=0.1, label='Homeostatic Zone')

    plt.title("Version 2: Stability Simplex and Coercion Dynamics")
    plt.savefig('results.png')
    print("[SUCCESS] Version 2 Results Saved to results.png")

if __name__ == "__main__":
    run_v2_simulation()
