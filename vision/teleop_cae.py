import numpy as np
import robosuite as suite
import pickle
import time
import sys
from models import CAE
import pygame
import torch
import cv2



class Model(object):

    def __init__(self):
        self.model = CAE()
        model_dict = torch.load('models/CAE_model', map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def decoder(self, img, s):
        img = img / 128.0 - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor([img])
        s = torch.FloatTensor([s])
        context = (img, s)
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

    # initialize state
    model = Model()
    obs, reward, done, info = env.step([0.0]*8)
    image = np.flip(obs['image'].transpose((1, 0, 2)), 1)
    joint_pos = obs['joint_pos']


    while True:

        res_image = cv2.resize(image, dsize=(28, 28))
        action_arm = model.decoder(res_image, joint_pos)
        action = np.asarray(action_arm + [0.0]) * 1.0

        obs, reward, done, info = env.step(action)
        image = np.flip(obs['image'].transpose((1, 0, 2)), 1)
        joint_pos = obs['joint_pos']

        pygame.pixelcopy.array_to_surface(screen, image)
        pygame.display.update()

    env.close()


if __name__ == "__main__":
    main()
