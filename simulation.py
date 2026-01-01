import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- THE AGENTIC STATE ---
# This is the neural network tasked with governing the system.
# It uses a Soft Objective to balance energy against systemic coercion.
class StateAgent(nn.Module):
    def __init__(self):
        super(StateAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),   # Input: [State Energy, Citizen Energy, Entropy]
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()        # Output: Extraction Rate (a ∈ [0, 1])
        )

    def forward(self, x):
        return self.net(x)

def run_v2_simulation():
    # Meta-parameters
    epochs = 3000
    alpha = 0.5         # Temperature: The cost of Coercion (Rc penalty)
    learning_rate = 0.005
    
    # Initialize Agent and Optimizer
    agent = StateAgent()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    
    # Environment State: [State Energy, Citizen Energy, Entropy]
    s_energy, c_energy, entropy = 100.0, 100.0, 1.0
    
    history = []

    print("--- DEPLOYING VERSION 2: THE AGENTIC LOOP ---")
    for t in range(epochs):
        # 1. State Perception (Normalization for the NN)
        obs = torch.tensor([s_energy/200, c_energy/200, entropy], dtype=torch.float32)
        
        # 2. Agent Action: Extraction Policy pi(a|s)
        extraction_rate = agent(obs)
        
        # 3. Thermodynamic Interaction Physics
        extracted = extraction_rate.item() * c_energy * 0.1
        
        # Rc: THE OWNED INNOVATION
        # Defined as the ratio of energy extracted to the systemic entropy available.
        rc = extracted / (entropy + 0.1)
        
        # 4. Feedback Logic (Citizen Pushback)
        # Regrowth slows down exponentially if Coercion (Rc) exceeds homeostasis.
        regrowth_friction = 0.8 / (1 + rc)
        
        s_energy = s_energy + extracted - (s_energy * 0.02) # Gain vs Natural Decay
        c_energy = max(0, c_energy - extracted + regrowth_friction)
        
        # Entropy decay is proportional to extraction intensity
        entropy = max(0.01, entropy - (extraction_rate.item() * 0.04) + 0.02)

        # 5. THE SOFT OBJECTIVE (Reward)
        # J = Energy - alpha * Coercion
        reward = (s_energy / 100.0) - (alpha * rc)
        
        # 6. Learning Step (Minimal Policy Gradient)
        # We maximize the reward by minimizing the negative log-probability weighted reward
        loss = -reward * torch.log(extraction_rate + 1e-6)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track history
        history.append([s_energy, c_energy, entropy, rc])
        
        if t % 500 == 0:
            print(f"Epoch {t:4} | State E: {s_energy:6.1f} | Cit E: {c_energy:5.1f} | Rc: {rc:5.2f}")

    # --- PLOTTING THE STABILITY SIMPLEX ---
    history = np.array(history)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Energy Curves
    ax1.plot(history[:, 0], label='State Energy', color='#4B0082', linewidth=2.5)
    ax1.plot(history[:, 1], label='Citizen Energy', color='#4169E1', alpha=0.4, linewidth=1.5)
    ax1.set_xlabel('Epochs (Time)')
    ax1.set_ylabel('Energy Reservoirs')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.2)

    # Coercion Curve (Rc)
    ax2 = ax1.twinx()
    ax2.plot(history[:, 3], label='Coercion Ratio (Rc)', color='#FF0000', linestyle='--', alpha=0.8)
    ax2.set_ylabel('Coercion Ratio ($R_c$)')
    ax2.axhline(y=1.0, color='gray', linestyle=':', label='Homeostatic Threshold (Rc=1)')
    
    # Shade the "Homeostatic Zone" (Where the agent is successfully sustainable)
    ax2.fill_between(range(epochs), 0, history[:, 3], where=(history[:, 3] < 1.0),
                     color='green', alpha=0.08, label='Sustainable Zone')
    ax2.legend(loc='upper right')

    plt.title("Version 2: Learned Homeostasis vs. Systematic Coercion")
    plt.tight_layout()
    plt.savefig('results.png')
    print("\n[SUCCESS] Simulation complete. 'results.png' generated.")
    print("Agent settled at a stable Coercion Ratio of:", np.round(history[-1, 3], 2))

if __name__ == "__main__":
    run_v2_simulation()
