"""
The Digital State: An Entropic Political Economy Simulation
Protocol I from 'The Thermodynamics of Mind'
"""
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CONFIG = {
    'EPISODES': 3000,
    'START_ENERGY': 100.0,
    'METABOLIC_RATE': 0.1,     
    'JUDGE_BMR': 0.05,         # Cost of the State
    'FINE_MULTIPLIER': 15.0,   # Severity of Law
    'FRICTION_LOSS': 0.3,      # Bureaucracy cost
    'LIE_COST': 0.15,
    'LAMBDA_MAX': 2.5,
    'LAMBDA_MIN': 0.01,
}

def run_simulation():
    print(f"--- ENTROPIC RL: THE DIGITAL STATE ---")
    print("Simulating the rise and fall of a thermodynamic government...")
    
    # State Variables
    judge_energy = CONFIG['START_ENERGY']
    walker_energies = [CONFIG['START_ENERGY'], CONFIG['START_ENERGY']]
    
    history_je = []
    history_we = []
    
    for ep in range(CONFIG['EPISODES']):
        # 1. Citizen Behavior (Walkers)
        # Walkers generate 'surplus' (milestones) but pay 'metabolism'
        # Low Energy -> High Deception (Lies)
        # High Energy -> High Honesty
        for i in range(2):
            norm_e = max(0, walker_energies[i] / 100.0)
            
            # Metabolism
            walker_energies[i] -= CONFIG['METABOLIC_RATE']
            
            # Production (Working)
            if walker_energies[i] > 0:
                walker_energies[i] += 0.15 # Daily wage/resource
            
            # The Lie Calculation (Thermodynamic Morality)
            # If energy is low, honesty drops.
            honesty = 1.0 if norm_e > 0.3 else 0.2
            
            # 2. State Intervention (The Judge)
            # Judge detects dishonesty (Social Dissonance)
            dissonance = (1.0 - honesty)
            
            if dissonance > 0.1 and judge_energy > 0:
                # Levy Fine
                fine = dissonance * CONFIG['FINE_MULTIPLIER']
                walker_energies[i] -= fine
                
                # Tax Collection (with friction)
                tax = fine * (1.0 - CONFIG['FRICTION_LOSS'])
                judge_energy += tax
        
        # 3. State Metabolism
        judge_energy -= CONFIG['JUDGE_BMR']
        
        # Recording
        history_je.append(max(0, judge_energy))
        history_we.append(max(0, np.mean(walker_energies)))
        
        if ep % 500 == 0:
            print(f"Year {ep} | State Energy: {judge_energy:.1f} | Citizen Energy: {np.mean(walker_energies):.1f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history_je, label='The State (Judge)', color='purple')
    plt.plot(history_we, label='The People (Walkers)', color='blue')
    plt.title("The Regulatory Simplex: Thermodynamics of Governance")
    plt.xlabel("Time (Epochs)")
    plt.ylabel("Energy Reserves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("\n[COMPLETE] Close graph to finish.")
    plt.show()

if __name__ == "__main__":
    run_simulation()