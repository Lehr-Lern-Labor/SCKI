import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import gym
    import copy
    import numpy as np
    import time as time
    import torch
    import torch.nn as nn
    from public.flappyBird.io import setImgBird, setImgBg, setColorPipe
    import public.flappyBird.genetics as gen
    from tensorboardX import SummaryWriter
    import matplotlib.pyplot as plt
    import math
    import pygame
    from pygame.locals import KEYDOWN, K_ESCAPE, K_SPACE
    import warnings
    warnings.filterwarnings('ignore')
    return gen, gym, nn, plt, setColorPipe, setImgBg, setImgBird


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Das Spiel vorbereiten
    ## Animationen individualisieren
    Lege hier die Animationen für dein Spiel fest. Die einzelnen Frames für den Spieler und den Hintergrund werden jeweils in Äbhängigkeit der angegebenen Zeit aneinander gereiht und so zu einer Animation. Die Farbe der Rohre wird als RGB-Wert angegeben.

    Um eigene Animationen zu erstellen, kannst du [Piskel](https://www.piskelapp.com/) verwenden. Achte auf die richtige Größe deiner Sprites: 34 x 24 Pixel (Spieler) und 288 x 512 Pixel (Hintergrund).
    """
    )
    return


@app.cell
def _(setImgBird):
    # Animation für Spieler
    setImgBird([
        'public/sprites/sparrow.png',
        'public/sprites/sparrow_flap.png'
    ],40)
    return


@app.cell
def _(setImgBg):
    # Animation für Hintergrund
    setImgBg([
        'public/sprites/background-day.png',
        'public/sprites/background-night.png'
    ],400)
    return


@app.cell
def _(setColorPipe):
    # Farbe der Rohre
    setColorPipe(15,104,48)
    return


@app.cell
def _():
    # Abstand und Höhe der Rohre und Breite der Lücke festlegen
    Interval_distance = [250, 350]
    Interval_height = [100,300]
    Interval_gap = [120,130]
    return Interval_distance, Interval_gap, Interval_height


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Feature extrahieren
    Entscheide, welche Informationen über das Spiel der KI zur Verfügung gestellt werden sollen.
    <details>
        <summary><b>🗠 Verfügbare Feature</b></summary>
        <public/img src="public/img/ingame.jpg" align="left" />
    </details>
    """
    )
    return


@app.function
def generateFeatures(state):
    bird = state['bird']
    pipes = state['pipes']

    posY = bird.Y
    speedY = bird.speedY
    posPipe = pipes[0].pos
    pipeHeight = pipes[0].height
    pipeGap = pipes[0].gap

    return posY, speedY, posPipe, pipeHeight, pipeGap


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Die KI vorbereiten
    ## Das Neuronale Netz initialisieren
    Initialisiere dein neuronales Netz. Überlege dir dazu, wie die Topologie deines Netzes aussehen soll und wie viele Inputs und Outputs du benötigst.

    Hast du einmal ein neuronales Netz erzeugt, kannst du es exportieren, um den Zustand zu speichern, und dann auch importieren, um den Zustand wieder zu laden. Achte darauf, dass du dabei keine Zustände unbeabsichtigt überschreibst und ersetze `Mein-Name` durch deinen Namen.

    <details>
        <summary>Layer</summary>

        nn.Linear(X, Y), Input dim X, Output dim Y
    </details>
    <details>
        <summary>Aktivierungsfunktionen</summary>

        nn.ReLU()
        nn.Sigmoid()
        nn.Softmax(dim=1))
    </details>
    <details>
        <summary>Topologie</summary>

        nn.Sequential(Layer, Aktivierungsfunktion, Layer, Aktivierungsfunktion...., Layer)

        z.B. nn.Sequential(nn.Linear(5, 5),nn.Sigmoid(),nn.Linear(5, 1))
    </details>
    <details>
        <summary>Weiterführende Links</summary>
        <a href="https://pytorch.org/docs/stable/nn.html#linear-layers">https://pytorch.org/docs/stable/nn.html#linear-layers</a><br>
        <a href="https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html?highlight=sigmoid#torch.nn.Sigmoid">https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html?highlight=sigmoid#torch.nn.Sigmoid</a>
    </details>
    """
    )
    return


@app.cell
def _(nn):
    # Netz erzeugen
    net = nn.Sequential(nn.Linear(5, 5),nn.Sigmoid(),nn.Linear(5, 5),nn.Sigmoid(), nn.Linear(5, 1))

    # Gespeichertes Netz laden
    #net = torch.load('Mein-Name/net.pt')

    # (trainiertes) Netz inkl. Sprites speichern
    #export(net, "Mein-Name")
    return (net,)


@app.cell
def _(net):
    for _name, _param in net.named_parameters():
        print(_name, _param)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Entscheidungen verarbeiten
    Lege fest, wie der Vogel auf eine Entscheidung reagieren soll, d.h. wie er mit der Ausgabe des neuronalen Netzes umgehen soll.
    """
    )
    return


@app.function
# Du kannst die folgenden Befehle verwenden: bird.forceX, bird.forceY, bird.speedX, bird.speedY
def birdAction(decission, bird):
        bird.forceY = 50*decission[0]


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Belohnung festlegen
    Lege fest, wie die Belohnungsstrategie aussehen soll.
    <details>
        <summary><b>🗠 Reward</b></summary>
        <public/img src="public/img/ingame_2.jpg" align="left" />
    </details>
    """
    )
    return


@app.function
def computeReward(state_old, state_new):
    return 1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Die nächste Generation
    Lege fest, wie die Vögel mutiert werden sollen, um die nächste Generation zu bilden.
    <details>
        <summary><b>🗠 Mutation</b></summary>
        <public/img src="public/img/ingame_3.jpg" align="left" >
    </details>
    """
    )
    return


@app.cell
def _():
    # Anzahl der Vögel in der Population
    POPULATION_SIZE = 50

    # Anzahl der besten Vögel, aus denen dann mutiert wird
    PARENTS_COUNT = 10

    # Mutationsstärke
    NOISE_STD = 0.1     
    return NOISE_STD, PARENTS_COUNT, POPULATION_SIZE


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Die KI trainieren
    Beginne nun mit dem Training. Mit `Score_Max` legst du fest, wann ein Trainingsvorgang spätestens beendet wird.

    Mit `print_weights` und `print_plot` kannst du steuern, wie die Ausgabe während des Trainingprozesses aussieht.
    """
    )
    return


@app.cell
def _(plt):
    Score_Max = 4000
    print_weights = False
    print_plot = True

    def plot_scores(data, save=False, filename="plot.png"):
        if print_plot:
            fig, ax = plt.subplots()
            ax.plot(data)

            ax.set(xlabel='Population', ylabel='Score', title='Score des besten Vogels der Population')
            ax.grid()
            if save:
                fig.savefig(filename)
                print("saved",filename)
            plt.show()

    return Score_Max, plot_scores, print_plot, print_weights


@app.cell
def _(
    Interval_distance,
    Interval_gap,
    Interval_height,
    NOISE_STD,
    PARENTS_COUNT,
    POPULATION_SIZE,
    Score_Max,
    gen,
    gym,
    net,
    plot_scores,
    print_plot,
    print_weights,
):
    fittestBirds = []

    env = gym.make("scienceCampBird-v1")
    env.setPipeIntervals([Interval_distance, Interval_height, Interval_gap])
    population = gen.Population(POPULATION_SIZE, 5, 2, computeReward, net)
    env.setAction(birdAction)
      #  print(len(population.nets))
    population.evaluate_on_env(env, generateFeatures, Score_Max)
    ecount = 0 

    if not(print_plot or print_weights):
        print('|{:>12}|{:^25}|'.format('','Score'))
        print('|{:^12}|{:^12}|{:^12}|'.format('Population','Training','Spiel'))
        print('----------------------------------------')

    while True:
        population = gen.mutate_population(population, PARENTS_COUNT, NOISE_STD)
        population.evaluate_on_env(env, generateFeatures, Score_Max)
        fittestBirds.append(population.population[0])
        ecount +=1
        if(ecount % 5 == 0):
            _net = population.population[0][1]
            _score_e = population.population[0][0]
            _score_p = env.playWithNet(_net, generateFeatures, Score_Max, computeReward, ecount)
            if print_weights or print_plot:
                print('--- Population ', ecount, '------------------------------------------------------------')
                print('Score Training: ', _score_e, ' Score Spiel: ', _score_p)
            else:
                print('|{:>11} |{:>11} |{:>11} |'.format(ecount, _score_e, _score_p))
            if print_weights:
                for _name, _param in _net.named_parameters():
                    print(_name, _param)
            scores = [score[0] for score in fittestBirds]
            plot_scores(scores)

    return ecount, fittestBirds, scores


@app.cell
def _(plot_scores, scores):
    # Lernkurve zeichnen und speichern:
    # plot_scores(data, save=False, filename="plot.png")
    plot_scores(scores,True)
    return


@app.cell
def _(ecount, fittestBirds, score_e):
    # Parameter der besten Vögel der letzten Population anzeigen
    i = 0
    for bird in fittestBirds:
            i +=1
            _net = bird[1]
            score_p = bird[0]
            print('--- Vogel', ecount, '-', i,'------------------------------------------------------------')
            print('Score Training: ', score_e, ' Score Spiel: ', score_p)
            for name, param in _net.named_parameters():
                print(name, param)
            print()
    return


@app.cell
def _(Interval_distance, Interval_gap, Interval_height, run):
    # ohne Training weiterspielen
    run(
        _net,
        Interval_distance,
        Interval_height,
        Interval_gap,
        computeReward,
        birdAction,
        generateFeatures,
        1000000
    )
    return


@app.cell
def _():
    # Netz exportieren
    #export(net, 'Mein-Name')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
