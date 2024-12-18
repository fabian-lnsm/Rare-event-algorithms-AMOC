import matplotlib.pyplot as plt
import numpy as np

# Data parsed from the provided file
initial_states = {
    "[ 4.         -0.93351693]": {
        "decay_lengths": [0.2, 0.5, 0.8, 2.0, 5.0, 10.0],
        "probabilities": [2.162300836203422e-05, 1.726396680024552e-05, 1.7137620940131387e-05, 1.6496166139716303e-05, 1.7483373628186453e-05, 1.6339990871216072e-05],
        "stddevs": [1.4910499633124708e-05, 2.8882512629414356e-06, 1.8405458102843246e-06, 2.216963108138017e-06, 2.78372887823317e-06, 2.3580313555600947e-06]
    },
    "[ 7.         -0.87117525]": {
        "decay_lengths": [0.5, 0.8, 2.0, 5.0, 10.0, 30.0],
        "probabilities": [0.0007567694935329913, 0.000753320858584551, 0.0006548559597307684, 0.0007230560356567757, 0.0007729850103260532, 0.0007627612367616139],
        "stddevs": [9.680600699713979e-05, 8.243571776558859e-05, 5.321737320872146e-05, 6.801897162606967e-05, 9.30573127085743e-05, 7.917964331653308e-05]
    },
    "[10.         -0.78648254]": {
        "decay_lengths": [0.2, 0.8, 5.0, 10.0, 30.0],
        "probabilities": [0.012386554142108926, 0.011337305417939786, 0.011052087435989994, 0.009655680018330389, 0.010968422112554065],
        "stddevs": [0.0017250244994230053, 0.0009357629920318089, 0.0007902600975269483, 0.0009042520722479226, 0.0008266948839658793]
    },
}

# MC Results for horizontal lines
mc_results = [
    (1.5e-5, 0.3e-5),  # (value, uncertainty) for state 1
    (6.7e-4, 0.7e-4),  # for state 2
    (9.5e-3, 0.3e-3)   # for state 3
]

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

for ax, ((state, data), mc_result) in zip(axs, zip(initial_states.items(), mc_results)):
    # Plot error bars for data
    ax.errorbar(data["decay_lengths"], data["probabilities"], yerr=data["stddevs"], fmt='o', label=f"Initial state: {state}")
    
    # Plot horizontal line for MC result
    mc_value, mc_uncertainty = mc_result
    ax.axhline(mc_value, color='red', linestyle='--', label="MC result")
    ax.fill_between([min(data["decay_lengths"]), max(data["decay_lengths"])], 
                    mc_value - mc_uncertainty, mc_value + mc_uncertainty, 
                    color='red', alpha=0.2, label="MC uncertainty")

    ax.set_ylabel("Probability (avg ± stddev)")
    ax.legend()
    ax.grid(True)

axs[0].set_xlabel("Decay Length")
axs[1].set_xlabel("Decay Length")
axs[2].set_xlabel("Decay Length")

# Adjust layout and display the plot
plt.tight_layout()

plt.savefig('../temp/decay_length.png')
