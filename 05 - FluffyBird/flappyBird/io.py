import pygame
import torch
import os
import gym
import flappyBird.genetics as gen

img_bird = ['sprites/sparrow.png','sprites/sparrow_flap.png']
tick_bird = 8
img_bg = ['sprites/background-day.png']
tick_bg = 8
color_pipe = pygame.Color(15,104,47)

def setImgBird(img, tick = 8):
    global img_bird
    img_bird = img
    global tick_bird
    tick_bird = tick
    
def getImgBird():
    return img_bird

def getTickBird():
    return tick_bird
    
def setImgBg(img, tick = 8):
    global img_bg
    img_bg = img
    global tick_bg
    tick_bg = tick
    
def getImgBg():
    return img_bg

def getTickBg():
    return tick_bg
    
def setColorPipe(r,g,b,a = 255):
    global color_pipe
    color_pipe = pygame.Color(r,g,b,a)

def getColorPipe():
    return color_pipe

def export(net,name):
    # NN exportieren
    if not os.path.exists(name):
        os.mkdir(name)
    torch.save(net, name + "/net.pt")
    
    files_bird = []
    files_bg = []
    
    for img in img_bird:
        file = name + "/" + img.split('/')[-1]
        os.system('cp ' + img + ' ' + file)
        files_bird.append(file)
        
    for img in img_bg:
        file = name + "/" + img.split('/')[-1]
        os.system('cp ' + img + ' ' + file)
        files_bg.append(file)
        
    f = open(name + "/setup.txt", "w")
    f.write("setImgBird(" + str(files_bird) + ", " + str(tick_bird) + ")" "\n")
    f.write("setImgBg(" + str(files_bg) + ", " + str(tick_bg) + ")" "\n")
    f.write("setColorPipe" + str(color_pipe))
    f.close()
    
def run(net,Interval_distance, Interval_height,Interval_gap,computeReward,birdAction,generateFeatures,Score_Max):
    try:
        env = gym.make("scienceCampBird-v1")
        env.setPipeIntervals([Interval_distance, Interval_height,Interval_gap])
        env.setAction(birdAction)
        ecount = 0
        while True:
            env.playWithNet(net, generateFeatures, Score_Max, computeReward, ecount)
    except pygame.error:
        return
    