import numpy as np
import robosuite as suite
import pickle
import time
import sys
from models import CAE
import pygame
import torch
import cv2


class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1
        self.SCALE = 5.0

    def input(self):
        pygame.event.get()
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        if abs(z1) < self.DEADBAND:
            z1 = 0.0
        reset = self.gamepad.get_button(7)
        return [z1 * self.SCALE], reset


class Model(object):

    def __init__(self):
        self.model = CAE()
        model_dict = torch.load('models/CAE_model', map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def decoder(self, img, s, z):
        img = img / 128.0 - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor([img])
        s = torch.FloatTensor([s])
        z = torch.FloatTensor([z])
        context = (img, s, z)
        a_tensor = self.model.decoder(context)
        a_numpy = a_tensor.detach().numpy()[0]
        return list(a_numpy)



def main():

    width = 512
    height = 512
    screen = pygame.display.set_mode((width, height))

    # create environment instance
    env = suite.make(
        "PandaILIAD",
        has_renderer=False,
        ignore_done=True,
        camera_height=height,
        camera_width=width,
        gripper_visualization=False,
        use_camera_obs=True,
        use_object_obs=False,
    )

    # reset the environment
    env.reset()

    # create the human input device
    joystick = Joystick()
    model = Model()
    action_scale = 0.1

    # initialize state
    obs, reward, done, info = env.step([0.0]*8)
    image = np.flip(obs['image'].transpose((1, 0, 2)), 1)
    joint_pos = obs['joint_pos']


    while True:

        # get human input
        z, reset = joystick.input()
        if reset:
            return True

        # get observation
        res_image = cv2.resize(image, dsize=(28, 28))

        # decode to high-DoF action
        action_arm = model.decoder(res_image, joint_pos, z)
        action = np.asarray(action_arm + [0.0]) * action_scale

        # take action
        obs, reward, done, info = env.step(action)
        image = np.flip(obs['image'].transpose((1, 0, 2)), 1)
        joint_pos = obs['joint_pos']

        pygame.pixelcopy.array_to_surface(screen, image)
        pygame.display.update()

    env.close()


if __name__ == "__main__":
    main()
