import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Repräsentation von Informationen im Computer

        Nachdem du einige Grundlagen der Pythonprogrammierung kennen gelernt hast, widmen wir uns in diesem Abschnitt der Repräsentation von Informationen im Computer, die für die Anwendungen künstlicher Intelligenz besonders wichtig ist. 

        Um es möglichst einfach zu halten, betrachten wir zwei Arten von Gegenständen: die Eier von Blaumeisen und die Eier von Greifvögeln. 

        &nbsp;


         <figure>
          <img src="public/img/voegel_und_eier.png" alt="Vögel und Eier" style="width:45%">
          <figcaption></figcaption>
        </figure> 

        &nbsp;

        Diese Objekte unterscheiden sich z.B. in ihrer Helligkeit und Höhe voneinander. Die Eier der Blaumeisen sind relativ hell und klein. Demgegenüber sind die Eier mancher Greifvögel besonders hoch und relativ dunkel. Messen wir die Helligkeit und die Höhe eines Eis einer Blaumeise und eines Greifvogels, so könnte z.B. der zweidimensionale Punkt $p_1=(0.8, 2)$ ein Ei einer Blaumeise und $p_2=(0.3, 8)$ ein Ei eines Greifvogels beschreiben. Dabei steht der erste Wert für die Helligkeit und der zweite für die Höhe des Eis. Demzufolge können wir jedes Ei als Punkt in einem Koordinatensystem darstellen.  

        &nbsp;


         <figure>
          <img src="public/img/koordinatensystem_eier.png" alt="Koordinatensystem mit Eiern" style="width:45%">
          <figcaption></figcaption>
        </figure> 

        &nbsp;

        Für Computer sind also derartige Informationen nichts weiter als zwei-, drei- oder n-dimensionale Datenpunkte, denn ein Computer hat im Gegensatz zum Menschen kein intuitives Verständnis von Formen, Farben oder sonstigen Sinneswahrnehmungen. 

        ## Darstellung von Datenpunkten in Python

        Um Datenpunkte in Python darzustellen, gibt es unterschiedliche Bibliotheken, wie z.B. <i>NumPy</i> oder <i>PyTorch</i>. Für den jetzigen Anwendungsfall reicht es jedoch aus, einen Datenpunkt durch ein Zahlentupel zu beschreiben. Möchtest du diese Datenpunkte in einem zweidimensionalen Koordinatensystem abbilden, bietet dir die Bibliothek <i>Matplotlib</i> die passende Funktion dazu.
        """
    )
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt

    # drei zweidimensionale Datenpunkte
    d1 = (1, 2)
    d2 = (-1, 1)
    d3 = (0, 2)

    datenpunkte = [d1, d2, d3]

    # Um diese Datenpunkte in einem Koordinatensystem darzustellen,
    # musst du die x- und y-Werte voneinander trennen.
    # Wie du in der Einführungseinheit gelernt hast, hast du dafür zwei Möglichkeiten.
    # Entweder du verwendest eine for-Schleife und fügst das erste bzw. zweite Element
    # in die entsprechende Liste hinzu....
    x_werte = []
    y_werte = []
    for datenpunkt in datenpunkte:
        x_werte.append(datenpunkt[0])
        y_werte.append(datenpunkt[1])
    
    print("for-Schleife:")    
    print("x-werte:", x_werte)
    print("y-werte:", y_werte)

    # ... oder du verwendest list comprehension,
    # um die Listen zu erstellen.
    x_werte = [datenpunkt[0] for datenpunkt in datenpunkte]
    y_werte = [datenpunkt[1] for datenpunkt in datenpunkte]

    print("\nlist comprehension:")
    print("x-werte:", x_werte)
    print("y-werte:", y_werte)

    # Nun kannst du dir die Punkte mit folgender Anweisung plotten lassen.
    plt.scatter(x_werte, y_werte)

    # Hast du einen Punkt vergessen, kannst du ihn nachträglich in das 
    # Koordinatensystem eintragen. Die Farbe der Punkte kannst du mit 
    # dem Parameter 'color' festlegen.
    plt.scatter([1], [1], color='red')
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <img style="float: left;" src="public/img/laptop_icon.png" width=50 height=50 /> <br><br>

        <i>Plotte die Punkte des Graphen der Funktion $x^2$ im Koordinatensystem, deren x-Werte im Intervall von $-4$ bis $4$ liegen und jeweils den Abstand $0.5$ voneinander haben.</i>
        """
    )
    return


@app.cell
def _():
    # Füge hier deinen Code ein.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## k-Nearest-Neighbors (kNN)

        Im Folgenden schauen wir uns unseren ersten KI-Algorithmus an. Der k-Nearest-Neighbor-Algorithmus ist ein einfacher Klassifikationsalgorithmus. Im ersten Schritt wird dem Computer „mitgeteilt“, bei welchen Daten es sich um Datenpunkte der Klasse 0, 1, 2, etc. handelt; in unserem Fall welche Datenpunkte die Eier der Blaumeise und welche Datenpunkte die Eier von Greifvögeln repräsentieren. Im zweiten Schritt wird <i>k</i> festgelegt. Wir wählen $k=3$. Wenn wir nun einen neuen Datenpunkt klassifizieren müssen, dessen Klasse uns nicht bekannt ist, bestimmen wir die $k=3$ nächsten Nachbarn dieses Datenpunkts. Hat dieser Datenpunkt mehr Nachbarn, die als Eier der Blaumeise klassifiziert sind, wird auch dieser Datenpunkt als ein Ei der Blaumeise klassifiziert. Gibt es mehr Eier der Greifvögel als Nachbarn, wird der neue Datenpunkt als Ei eines Greifvogels klassiziert. Die Herausforderung des k-Nearest-Neighbours-Algorithmus besteht u.a. darin, <i>k</i> optimal zu wählen.

        &nbsp;


         <figure>
          <img src="public/img/koordinatensystem_eier_knn.png" alt="Koordinatensystem mit Eiern" style="width:45%">
          <figcaption></figcaption>
        </figure> 

        &nbsp;

        <img style="float: left;" src="public/img/laptop_icon.png" width=50 height=50 /> <br><br>

        <i> Jetzt betrachten wir Mini-Greifvögel, deren Eier etwas kleiner und heller sind als die der bisherigen Greifvögel. Außerdem haben wir ein paar unbekannte Eier (rot markierte Datenbpunkte), die entweder von den Mini-Greifvögeln oder den Blaumeisen stammen. Ordne die unbekannten Eier mit Hilfe des kNN-Algorithmus entweder der Klasse der Eier der Blaumeise oder der Eier der Greifvögel zu. Führe dazu das Feld aus, um alle Datenpunkte zu visualisieren. Implementiere anschließend den kNN-Algorithmus und führe die implementierte Funktion auf allen Datenpunkten aus. Ändert sich die Klassifikation für unterschiedliche <i>k</i>?</i>
        """
    )
    return


@app.cell
def _(plt):
    hoehe_eier_blaumeise = [1.7, 1.4, 1.75, 1.6, 1.8, 1.6, 1.4, 1.5, 2.0, 2.1]
    helligkeit_eier_blaumeise = [0.85, 0.9, 0.73, 0.66, 0.63, 0.57, 0.63, 0.65, 0.71, 0.75]

    hoehe_eier_mini_greifvogel = [2.4, 2.8, 2.35, 2.2, 2.9, 2.5, 2.2, 2.5, 3, 2.7]
    helligkeit_eier_greifvogel = [0.55, 0.6, 0.63, 0.7, 0.63, 0.50, 0.33, 0.45, 0.51, 0.55]

    hoehe_eier_unbekannt = [2.0, 2.2, 2.5, 2.9]
    helligkeit_eier_unbekannt = [0.5, 0.75, 0.7, 0.65]

    farbe_eier_blaumeise = '#ec90cc'
    farbe_eier_greifvogel = '#4f7087'



    plt.scatter(hoehe_eier_blaumeise, helligkeit_eier_blaumeise, color=farbe_eier_blaumeise)
    plt.scatter(hoehe_eier_mini_greifvogel, helligkeit_eier_greifvogel, color=farbe_eier_greifvogel)
    plt.scatter(hoehe_eier_unbekannt, helligkeit_eier_unbekannt, color='red')

    # Füge hier deinen Code ein.

    '''
    @param datenpunkte1: Alle Datenpunkte der Klasse 0
    @param datenpunkte2: Alle Datenpunkte der Klasse 1
    @param unbekannter_datenpunkt: Datenpunkt, der klassifiziert werden muss
    @param k: k gibt an, wie viele Nachbarn der kNN-Algorithmus bei seiner Entscheidung berücksichtigt
    @return: 0, wenn Ei der Blaumeise oder 1, wenn Ei eines Greifvogels
    '''
    def knn(datenpunkte1, datenpunkte2, unbekannter_datenpunkt, k):
        pass
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

