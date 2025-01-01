import matplotlib.pyplot as plt
import numpy as np


AMS_results_decaylengths = {
    "[ 4.         -0.93351693]": {
        "decay_lengths": [0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.4, 5.0],
        "probabilities": [
        1.7509699578735116e-05, 
        1.4019106061573621e-05, 
        1.821361884650993e-05, 
        1.7731277909538667e-05, 
        1.772120460075517e-05, 
        1.6922441903924316e-05, 
        1.6462248504867613e-05, 
        1.8652547971354358e-05
        ],
        "stddevs": [
        3.0965409676439208e-06, 
        1.5229962036679426e-06, 
        2.1342952952999365e-06, 
        2.659781951644599e-06, 
        1.920948950896054e-06, 
        2.6112784844069313e-06, 
        2.0872235595128045e-06, 
        2.2682656923468642e-06
        ]
    },
    "[ 7.         -0.87117525]": {
        "decay_lengths": [0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.4, 5.0],
        "probabilities": [
            0.0007478644197659352, 
            0.0007257794205572925, 
            0.0006584937350316939, 
            0.0006999467682192444, 
            0.0006638290531641145, 
            0.0007097333774945665, 
            0.0007312849587710725, 
            0.000782449609818967
        ],
        "stddevs": [
        9.596048358582488e-05, 
        8.514707082546096e-05, 
        7.031799440318032e-05, 
        6.16613638508171e-05, 
        8.582926363539744e-05, 
        6.66309542033023e-05, 
        9.428759011516106e-05, 
        7.262382578537882e-05
        ]
    },
    "[10.         -0.78648254]": {
        "decay_lengths": [0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.4, 5.0],
        "probabilities": [0.010366835055434778, 
        0.010911010517464683, 
        0.01114655380938673, 
        0.010306089023132441, 
        0.010213285813557672, 
        0.01089217413554363, 
        0.010480886530098836, 
        0.010359699378777823
        ],
        "stddevs": [0.0007940902903629793, 
        0.0008281207287413626, 
        0.0008907160816835, 
        0.0007375553812147127, 
        0.0006824650637110241, 
        0.0010728337150069799, 
        0.0008478265336334102, 
        0.0007656834041563668
        ]
    },
}

# optimized AMS results
AMS_results_optimized = [
    (1.5, 1.61346312308819e-5, 7.5039472683302423e-7), # for t_init=4.0
    (0,0,0), # for t_init=7.0
    (0,0,0) # for t_init=10.0
]

# MC Results
mc_results = [
    (1.5e-5, 0.3e-5),  # for state t_init=4.0
    (6.7e-4, 0.7e-4),  # for state t_init=7.0
    (9.5e-3, 0.3e-3)   # for state t_init=10.0
]

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

for ax, ((init_state, AMS_decayLength), mc_result, AMS_optim) in zip(axs, zip(AMS_results_decaylengths.items(), mc_results, AMS_results_optimized)):
    # Plot error bars for data
    ax.errorbar(
        AMS_decayLength["decay_lengths"], AMS_decayLength["probabilities"], yerr=AMS_decayLength["stddevs"],
                fmt='o', capsize=5, color='darkblue',
                label=f"Initial state: {init_state}"
        )
    
    # add precise results
    AMS_optim_decay, AMS_optim_value, AMS_optim_uncertainty = AMS_optim
    ax.errorbar(
        AMS_optim_decay, AMS_optim_value, yerr=AMS_optim_uncertainty,
        fmt='o', capsize=5, color='dodgerblue',
    )
    
    # Plot horizontal line for MC result
    mc_value, mc_uncertainty = mc_result
    ax.axhline(mc_value, color='red', linestyle='--', label="MC result")
    ax.fill_between([min(AMS_decayLength["decay_lengths"]), max(AMS_decayLength["decay_lengths"])], 
                    mc_value - mc_uncertainty, mc_value + mc_uncertainty, 
                    color='purple', alpha=0.2, label="MC uncertainty")
    

    ax.set_ylabel("Probability (avg ± stddev)")
    ax.legend()
    ax.grid(True)

axs[0].set_xlabel("Decay Length")
axs[1].set_xlabel("Decay Length")
axs[2].set_xlabel("Decay Length")

# Adjust layout and display the plot
plt.tight_layout()

plt.savefig('../temp/decay_length.png')
