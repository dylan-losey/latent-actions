import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.wrappers.ik_wrapper import IKWrapper
import pickle
import sys
import pygame
import time


class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.mode = 'trans'
        self.grasp = [0.0]
        self.DEADBAND = 0.1
        self.SCALE_TRANS = 0.0025
        self.SCALE_ROT = 0.02

    def input(self):
        pygame.event.get()
        # collect inputs
        e_stop = self.gamepad.get_button(7)
        z_open = self.gamepad.get_button(1)
        z_close = self.gamepad.get_button(0)
        z1 = self.gamepad.get_axis(1)
        z2 = self.gamepad.get_axis(0)
        z3 = -self.gamepad.get_axis(4)
        z = [z1, z2, z3]
        # determine control mode
        if self.mode == 'trans' and self.gamepad.get_button(4):
            self.mode = 'rot'
        elif self.mode == 'rot' and self.gamepad.get_button(5):
            self.mode = 'trans'
        for idx in range(len(z)):
            if abs(z[idx]) < self.DEADBAND:
                z[idx] = 0.0
        # map inputs to changes in end-effector
        dpos = np.array([0.0, 0.0, 0.0])
        dquat = np.array([0.0, 0.0, 0.0, 1.0])
        if self.mode == 'trans':
            dpos = np.asarray(z)*self.SCALE_TRANS
        elif self.mode == 'rot':
            dquat[0:3] = np.asarray(z)*self.SCALE_ROT
        if z_open:
            self.grasp = [0.0]
        elif z_close:
            self.grasp = [1.0]
        return dpos, dquat, self.grasp, e_stop


def main():

    # create environment instance
    env = suite.make(
        "PandaLift",
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        control_freq=100
    )

    # reset the environment
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # enable controlling the end effector directly instead of using joint velocities
    env = IKWrapper(env)

    # create the human input device
    joystick = Joystick()

    # create recorder
    demo_number = sys.argv[1]
    savename = "demonstrations/demo" + demo_number + ".pkl"
    sample_time = 0.1
    prev_time = time.time()
    data = []

    while True:

        # get human input
        dpos, dquat, grasp, reset = joystick.input()
        if reset:
            pickle.dump(data, open(savename, "wb"))
            break

        # take action and record joint position
        action = np.concatenate([dpos, dquat, grasp])
        obs, reward, done, info = env.step(action)
        joint_pos = obs['joint_pos']

        # save joint position to dataset
        curr_time = time.time()
        if curr_time - prev_time > sample_time:
            prev_time = curr_time
            data.append(list(joint_pos))
            pickle.dump(data, open(savename, "wb"))

        env.render()

    env.close()


if __name__ == "__main__":
    main()
