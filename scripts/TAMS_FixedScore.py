# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class TAMS_prescribed:
    """
    Class implementing the TAMS algorithm for prescribed score functions.
    """

    def __init__(self, N_traj, Tmax, model, scorefunc, params=None, seed=None):
        """
        Parameters
        ----------
        N : int
            Number of trajectories.
        Tmax : int
            Maximum length of trajectories.
        model : class
            Model to use.
        scorefunc : class
            Score function to use.
        params : dict, optional
            Parameters of the model.
        seed : int, optional
            Seed for the random number generator.
        """
        if type(Tmax) != int:
            raise ValueError("Tmax must be an integer.")
        self.model, self.scorefunc = model, scorefunc
        self.params = params
        self.N, self.Tmax = N_traj, Tmax
        self.rng = np.random.default_rng(seed=seed)

    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state

    def run(self, ic: np.ndarray, nc: int):
        """
        The TAMS algorithm.

        Parameters
        ----------
        ic : np.ndarray of shape (2,)
            Initial condition for each trajectory. Must be the same for every trajectory.
        nc : int
            Number of clones deleted at each iteration.
        """

        # Initial weights
        k, w = 0, 1

        # Turn the initial condition into Nx2 array
        ic = np.tile(ic, (self.N, 1))

        # Give a warning if n_c and N are not compatible
        if nc > 0.1 * self.N:
            print("Warning: nc is quite large compared to N.")

        # Initial pool of trajectories
        traj = self.model.trajectory(
            self.N, self.Tmax, *self.params.values(), init_state=ic
        )

        # Initial number of timesteps
        Nt = traj.shape[1]
        nb_total_timesteps = Nt * self.N

        # Initial score
        # Set to 0 for the on-states, to 1 for the off-states
        score = self.scorefunc.process(traj)
        onzone = self.model.is_on(traj)
        offzone = self.model.is_off(traj)
        score[onzone], score[offzone] = 0, 1

        # Maximum score for each trajectory
        Q = np.nanmax(score, axis=1)

        # Loop until you cannot discard nc trajectories anymore
        while len(np.unique(Q)) > nc:

            # Find the nc trajectories with the lowest score
            threshold = np.unique(Q)[nc - 1]  # Because Python counts from 0
            idx, other_idx = np.nonzero(Q <= threshold)[0], np.nonzero(Q > threshold)[0]
            Q_min = Q[idx]

            # Update weights
            w *= 1 - len(idx) / self.N

            # Create new trajectories
            new_ind = self.rng.choice(other_idx, size=len(idx))
            restart = np.argmax(score[new_ind] >= Q_min[:, np.newaxis], axis=1)
            init_clone = traj[new_ind, restart]

            # Update the number of timesteps
            all_length = Nt - restart
            nb_total_timesteps += np.sum(all_length)

            # Create new trajectories
            new_traj = self.model.trajectory(
                len(idx),
                int(self.Tmax * np.max(all_length) / Nt),
                *self.params.values(),
                init_state=init_clone,
            )

            # Update the score
            new_score = self.scorefunc.process(new_traj)

            # Update the trajectories and the score
            for i in range(len(idx)):
                t, r, l = idx[i], restart[i], all_length[i]

                traj[t, :r] = traj[new_ind[i], :r]
                traj[t, r:] = new_traj[i, :l]

                score[t, : r + 1] = score[new_ind[i], : r + 1]
                score[t, r + 1 :] = new_score[i, 1:l]

                onzone = self.model.is_on(traj[t])
                offzone = self.model.is_off(traj[t])

                score[t, onzone], score[t, offzone] = 0, 1

                # If some trajectories have hit the transition, they end with NaNs
                first_idx_off = np.argmax(offzone)
                if first_idx_off > 0:
                    score[t, first_idx_off + 1 :] = np.nan
                    traj[t, first_idx_off + 1 :] = np.nan

            k += 1

            Q = np.nanmax(score, axis=1)

        # Return the probability of the transition, the number of iterations, the number of trajectories that transitioned,
        #        the number of timesteps and the trajectories
        return (
            w * np.count_nonzero(Q >= 1) / self.N,
            k,
            np.count_nonzero(Q >= 1),
            nb_total_timesteps,
            traj,
        )

    def run_multiple(self, nb_runs, ic, nc, model):
        """
        Run the TAMS algorithm multiple times.

        Parameters
        ----------
        nb_runs : int
            Number of runs.
        ic : np.ndarray of shape (2,)
            Initial condition for each trajectory. Must be the same for every trajectory.
        nc : int
            Number of clones deleted at each iteration.

        Returns
        -------
        probabilities : np.ndarray of shape (nb_runs,)
            The transition probability for each run.
        """
        rng_model = np.random.default_rng(seed=23)
        seeds_model = rng_model.choice(100, size=nb_runs, replace=False)
        probabilities = np.zeros(nb_runs)
        for run in range(nb_runs):
            model.rng.bit_generator.state = np.random.PCG64(seeds_model[run]).state
            probability, _, _, _, _ = self.run(ic, nc)
            probabilities[run] = probability
        return probabilities


if __name__ == "__main__":

    from DoubleWell_Model import DoubleWell_1D
    from Score_Function import x_coord
    from plot_functions import plot_DoubleWell

    # Set up the simulation
    nc = 10
    nb_runs = 15
    N_traj = 1000
    tmax = 2
    dt = 0.01
    mu = 0.03
    noise_factor = 0.1
    filepath = "../temp/"
    filename = "simulationTAMS.txt"
    write_mode = "w"
    C_model = DoubleWell_1D()
    scorefunc = x_coord(C_model)

    # Print and save the simulation parameters
    print("-----------Simulation parameters------------------")
    print("Number of runs per initial condition: ", nb_runs)
    print("Number of trajectories: ", N_traj)
    print("Discarded per iteration: ", nc)
    print("Length of trajectories: ", tmax)
    print("Time step of the simulation: ", dt)
    print("Coupling parameter: ", mu)
    print("Noise factor: ", noise_factor)
    with open(filepath + filename, write_mode) as f:
        f.write(f"Nb_runs: {nb_runs} \n")
        f.write(f"N_traj: {N_traj} \n")
        f.write(f"n_disc: {nc} \n")
        f.write(f"T_max: {tmax} \n")
        f.write(f"dt: {dt} \n")
        f.write(f"mu: {mu} \n")
        f.write(f"noise_factor: {noise_factor} \n")

    ### run tams for Committor estimate
    #tmax10: initial_times = np.arange(3, 11, 0.2) #tmax10
    initial_times = np.arange(17, 19.1, 0.05) #tmax2
    #tmax10: initial_positions = np.arange(-1.0, -0.6, 0.01) #tmax10
    initial_positions = np.arange(-0.2, 0.25, 0.005) #tmax2
    T, P = np.meshgrid(initial_times, initial_positions)
    with tqdm(total=T.shape[0] * T.shape[1]) as pbar:
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                initial_condition = np.array([T[i, j], P[i, j]])
                #t_sim = int(tmax - initial_condition[0])
                t_sim = tmax #see if this works
                tams = TAMS_prescribed(
                    N_traj,
                    t_sim,
                    C_model,
                    scorefunc,
                    params={"dt": dt, "mu": mu, "noise": noise_factor},
                )
                probabilities = tams.run_multiple(
                    nb_runs, initial_condition, nc, C_model
                )
                with open(filepath + filename, "a") as f:
                    f.write(
                        f"{initial_condition[0]:.3f};{initial_condition[1]:.3f};{np.mean(probabilities):.10f};{np.std(probabilities):.10f}\n"
                    )
                pbar.update(1)
