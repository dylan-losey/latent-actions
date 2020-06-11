import pickle
import sys
import os
import numpy as np



def main():

    # size of window to take actions over
    n_step = int(sys.argv[1])

    # create dataset which will be filled with state-action pairs
    dataset = []
    folder = 'demonstrations'
    savename = 'data/traj_dataset.pkl'

    # for each demonstration
    for filename in os.listdir(folder):
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        n_states = len(traj)
        # for each start state
        for idx in range(n_step, n_states - n_step):
            # for each next state in the window
            for jdx in range(-n_step, n_step+1):
                s = np.array(traj[idx]) + np.random.normal(0, 0.1, 7)
                s = list(s)
                s1 = traj[idx + jdx]
                a = [s1[i] - s[i] for i in range(len(s))]
                # add state action pair to dataset
                dataset.append(s + a)

    # save and print for sanity check
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset)
    print(len(dataset))


if __name__ == "__main__":
    main()
