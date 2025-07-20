import marimo

__generated_with = "0.13.15"
app = marimo.App()


app._unparsable_cell(
    r"""
    import gym
    import copy
    import numpy as np
    import time as time
    import torch
    import torch.nn as nn
    from flappyBird.io import *
    import flappyBird.genetics as gen
    from tensorboardX import SummaryWriter
    import matplotlib.pyplot as plt
    import math
    import pygame
    from pygame.locals import *
    """,
    name="_"
)


@app.cell
def _(birdAction, computeReward, generateFeatures, run, torch):
    net = torch.load('Mein-Name/net.pt')

    def runDefault():
        run(
            net,
            [250, 350],        #Interval_distance
            [100, 300],        #Interval_height
            [120, 130],        #Interval_gap
            computeReward,
            birdAction,
            generateFeatures,
            10000              #Score_Max
        )
    return (runDefault,)


@app.cell
def _():
    def generateFeatures(state):
        bird = state['bird']
        posY = bird.Y
        speedY = bird.speedY
        pipes = state['pipes']
        return posY, speedY, pipes[0].pos-bird.Y, pipes[0].height, pipes[0].gap

    def birdAction(decission, bird):
            bird.forceY = 50 * decission[0]


    def computeReward(state_old, state_new):
        return 1
    return birdAction, computeReward, generateFeatures


@app.cell
def _(runDefault, setColorPipe, setImgBg, setImgBird):
    # DEFAULT -> setup.txt
    setImgBird(['public/sprites/sparrow.png', 'public/sprites/sparrow_flap.png'], 40)
    setImgBg(['public/sprites/background-night.png', 'public/sprites/background-day.png'], 400)
    setColorPipe(0, 150, 130)
    runDefault()
    return


if __name__ == "__main__":
    app.run()

