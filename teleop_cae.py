import numpy as np
import robosuite as suite
from train_model import CAE
import torch
import pygame


class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1
        self.SCALE = 1.0

    def input(self):
        pygame.event.get()
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        if abs(z1) < self.DEADBAND:
            z1 = 0.0
        if abs(z2) < self.DEADBAND:
            z2 = 0.0
        reset = self.gamepad.get_button(7)
        return [z1 * self.SCALE, z2 * self.SCALE], reset


class Model(object):

    def __init__(self):
        self.model = CAE()
        model_dict = torch.load('models/CAE_model', map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def decoder(self, z, q):
        if abs(z[0]) < 0.01:
             return [0.0] * 7
        z_tensor = torch.FloatTensor(z + q)
        a_tensor = self.model.decoder(z_tensor)
        return a_tensor.tolist()



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

    # create the human input device
    joystick = Joystick()
    model = Model()
    action_scale = 0.1

    # initialize state
    obs, reward, done, info = env.step([0.0]*8)
    s = list(obs['joint_pos'])


    while True:

        # get human input
        z, reset = joystick.input()
        if reset:
            return True

        # decode to high-DoF action
        action_arm = model.decoder(z, s)
        action = np.asarray(action_arm + [0.0]) * action_scale

        # take action
        obs, reward, done, info = env.step(action)
        s = list(obs['joint_pos'])
        env.render()


if __name__ == "__main__":
    main()
