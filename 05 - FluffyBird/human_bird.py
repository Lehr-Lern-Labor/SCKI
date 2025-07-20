import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import gym
    import copy
    import numpy as np
    import time as time
    import torch
    import torch.nn as nn
    import flappyBird
    import genetics as gen
    from tensorboardX import SummaryWriter
    import matplotlib.pyplot as plt
    import math
    import pygame
    from pygame.locals import K_ESCAPE, K_SPACE, KEYDOWN
    return KEYDOWN, K_ESCAPE, K_SPACE, gym, pygame, time


@app.function
def birdAction(decission, bird):
        bird.forceY = 400*decission[0]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## HIT SPACE TO FLAP""")
    return


@app.cell
def _(KEYDOWN, K_ESCAPE, K_SPACE, gym, pygame, time):
    done = False
    reward = 0
    env = gym.make("scienceCampBird-v1")
    state = env.reset()
    env.setPipeIntervals([[250,350], [100,300],[120,130]])
    env.setAction(birdAction)
    while True:
        decission = [0.0]     
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:

                pygame.quit()
            elif event.type == KEYDOWN and event.key == K_SPACE:
                decission = [1.0]

            #        print(acts)
        state_old = state
        state, _, done, _ = env.step(decission)
        reward += 1
        env.render()
        if done:
            state = env.reset()
            print ('Score:', reward)
            reward = 0
            time.sleep(1)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
