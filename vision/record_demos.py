import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.wrappers.ik_wrapper import IKWrapper
from robosuite.devices import Joystick
import pickle
import pygame
import sys
import cv2



def main():

    width = 512
    height = 512
    action_scale = 10.0
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

    # enable controlling the end effector directly instead of using joint velocities
    env = IKWrapper(env)

    # create the human input device
    joystick = Joystick()
    joystick.start_control()

    # create recorder
    number = sys.argv[1]
    savename = "demonstrations/test" + number + ".pkl"
    data = []


    while True:

        input = joystick.get_controller_state()
        dpos, dquat, grasp, reset = (
            input["dpos"],
            input["dquat"],
            input["grasp"],
            input["reset"],
        )
        if reset:
            pickle.dump(data, open(savename, "wb"))
            break

        action = np.concatenate([dpos, dquat, grasp]) * action_scale
        obs, reward, done, info = env.step(action)
        image = np.flip(obs['image'].transpose((1, 0, 2)), 1)
        joint_pos = obs['joint_pos']

        res_image = cv2.resize(image, dsize=(28, 28))
        data.append((res_image, joint_pos))

        pygame.pixelcopy.array_to_surface(screen, image)
        pygame.display.update()

    env.close()


if __name__ == "__main__":
    main()
