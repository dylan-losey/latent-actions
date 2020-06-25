import pickle
import sys
import os
import numpy as np



def main():

    dataset = []
    folder = 'demonstrations/'
    savename = 'data/traj_dataset.pkl'

    # for every demonstration
    for filename in os.listdir(folder):
        traj = pickle.load(open(folder + filename, "rb"))
        n_states = len(traj)
        print(filename, n_states)

        # get the goal image & robot state at goal
        goal = traj[-1]
        goal_s = goal[1]

        # for every state along path
        # get image and action to goal
        for idx in range(n_states):
            curr_item = traj[idx]
            image = curr_item[0] / 128.0 - 1.0
            image = np.transpose(image, (2, 0, 1))
            s = curr_item[1] + np.random.normal(0, 0.1, 7)
            a = goal_s - s
            dataset.append((image, s, a))

    pickle.dump(dataset, open(savename, "wb"))
    # print(dataset)
    print(len(dataset))


if __name__ == "__main__":
    main()
