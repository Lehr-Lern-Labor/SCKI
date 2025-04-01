import marimo

__generated_with = "0.10.18"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Das Perzeptron

        In der letzten Einheit haben wir gelernt, dass (reale) Objekte für den Computer nichts anderes als Datenpunkte sind und haben uns mit dem k-Nearest-Neighbors-Algorithmus beschäftigt. Im Folgenden betrachten wir unseren nächsten KI-Algorithmus, nämlich den Urgroßvater der heutigen neuronalen Netze: das Perzeptron.

        Importiere zunächst alle notwendigen Bibliotheken für dieses Jupyter Notebook, indem du das folgende Codefeld ausführst. Beachte, dass du dieses Codefeld bei jedem Neustart des Kernels erneut ausführen musst.
        """
    )
    return


@app.cell
def _():
    # Führe dieses Feld aus, indem du entweder oben auf 'Run' klickst oder 'Strg + Enter' drückst.
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    return np, plt, random, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h3>Motivation</h3>

        Das Perzeptron versucht die Funktionweise einer Nervenzelle, auch <b>Neuron</b> genannt, nachzubilden. Sehr vereinfacht gesagt, nimmt ein Neuron elektrische (und chemische) Signale auf, verarbeitet sie und leitet die verarbeiteten Signale weiter. 


        &nbsp;


         <figure>
          <img src="public/img/neuronale_zelle.png" alt="neuronale_zelle" style="width:50%">
          <figcaption></figcaption>
        </figure> 

        &nbsp;

        Bei der Nachbildung des biologischen Neurons brauchen wir zunächst Einheiten, die eine <b>Eingabe</b> aufnehmen können. Wir beschränken uns bei der Eingabe auf zwei Zahlen $x_1$ und $x_2$ (= Datenpunkt ($x_1$, $x_2$) im zweidimensionalen Raum). Diese Zahlen werden jeweils mit sogenannte <b>Gewichten</b> des Perzeptrons $w_1$ und $w_2$ multipliziert und zusammenaddiert. Zusätzlich addiert man noch den sogenannten <b>Bias</b> $b$ hinzu. Im nächsten Verarbeitungsschritt wird diese Summe in eine <b>Aktivierungsfunktion</b> eingesetzt. Als Aktivierungsfunktion wird in einem Perzeptron typischerweise eine spezielle <b>Treppenfunktion</b> verwendet. Wenn die Eingabe dieser Treppenfunktion kleiner oder gleich $0$ ist, wird $0$ zurückgegeben; wenn sie größer als $0$ ist, wird $1$ zurückgegeben:

        $$ f(x) = \left\{
        \begin{array}{ll}
        0 & x \leq 0 \\
        1 & \, \textrm{sonst} \\
        \end{array}
        \right. $$

        Die Ausgabe des Perzeptrons kann als Zuweisung der Eingabe zu einer bestimmten <b>Klasse</b> interpretiert werden, also der Klasse $0$ oder $1$. Das Perzeptron <b>klassifiziert</b> die eingegebenen Daten. Wenn wir an das Beispiel mit den Vogeleiern zurückdenken, so weist das Perzeptron im Optimalfall den Eiern der Blaumeisen den Wert / die Klasse 0 und den Eiern der Greifvögel den Wert / die Klasse 1 zu. 

        &nbsp;


         <figure>
          <img src="public/img/perzeptron.png" alt="perzeptron" style="width:70%">
          <figcaption></figcaption>
        </figure> 

        &nbsp;

        ____


        <img style="float: left;" src="public/img/laptop_icon.png" width=50 height=50 /> <br><br>

        <i>Setze das Perzeptron als Funktion in Python um. Rufe deine Funktion mit von dir gewählten Werten für $w_1$, $w_2$ und $b$ auf und überprüfe rechnerisch, ob die Ausgabe stimmt.</i>
        """
    )
    return


@app.cell
def _():
    _w1 = 0
    _w2 = 0
    _b = 0

    def perzeptron(x1, x2):
        pass
    return (perzeptron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In der letzten Aufgabe haben wir die Werte für $w_1$, $w_2$ und $b$ selbst festgelegt. Betrachten wir die Daten unserer Vogeleier, so möchten wir, dass das Perzeptron für einen Blaumeisen-Ei-Datenpunkt das Ergebnis 0 ausgibt und für einen Greifvogel-Ei-Datenpunkt das Ergebnis 1. Wahrscheinlich war unsere Wahl für die Werte $w_1$, $w_2$ und $b$ nicht optimal. Unser Ziel ist es, dass der KI-Algorithmus selbst die optimalen Werte für $w_1$, $w_2$ und $b$ des Perzeptrons bestimmt. Bevor wir untersuchen, wie das geht, klären wir, wie die uns zur Verfügung stehende Daten überlicherweise unterteilt werden.

        ## Trainings- und Testdaten

        Betrachten wir im Folgenden die uns zur Verfügung stehenden Daten, d.h. die Datenpunkte der Blaumeisen- und der Greifvögel-Eier.
        """
    )
    return


@app.cell
def _(plt):
    hoehe_eier_blaumeise = [2, 2.5, 2.7, 1.9, 3, 2.4, 2.8, 2.3, 3, 3.2]
    helligkeit_eier_blaumeise = [0.8, 0.75, 0.86, 0.82, 0.74, 0.9, 0.82, 0.87, 0.85, 0.81]

    hoehe_eier_greifvogel = [8.2, 8, 9, 7.4, 7.1, 8.7, 9.3, 8.6, 8.0, 8.5]
    helligkeit_eier_greifvogel = [0.1, 0.2, 0.24, 0.18, 0.21, 0.27, 0.28, 0.22, 0.35, 0.4]

    farbe_eier_blaumeise = '#ec90cc'
    farbe_eier_greifvogel = '#4f7087'

    plt.scatter(hoehe_eier_blaumeise, helligkeit_eier_blaumeise, color = farbe_eier_blaumeise)
    plt.scatter(hoehe_eier_greifvogel, helligkeit_eier_greifvogel, color = farbe_eier_greifvogel)
    return (
        farbe_eier_blaumeise,
        farbe_eier_greifvogel,
        helligkeit_eier_blaumeise,
        helligkeit_eier_greifvogel,
        hoehe_eier_blaumeise,
        hoehe_eier_greifvogel,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Um eine KI mit Daten zu trainieren, unterteilen wir die uns zur Verfügung stehende Daten in <b>Trainigsdaten</b> und <b>Testdaten</b>. Überlicherweise unterteilt man die Daten in 80% Trainings- und 20% Testdaten, also ein Verhältnis von 8:2. Auf den Trainigsdaten wird die KI trainiert, d.h. sie passt sich so an, dass sie viele Trainingsdaten richtig klassifiziert. In unserem Fall bedeutet es, dass sie einen Trainingspunkt $(2, 0.8)$ als Ei einer Blaumeise klassifiziert. Nach einem Trainingsdurchlauf (auch <b>Epoche</b> genannt) wird auf den Testdaten die Genauigkeit (engl. Accuracy) der KI gemessen. Es ist dabei wichtig, dass es bei den Trainings- und Testdaten keine Überschneidungen gibt!

        <img style="float: left;" src="public/img/laptop_icon.png" width=50 height=50 /> <br><br>

        <i>Erstelle eine Liste mit allen Datenpunkten als ($x_1$, $x_2$, $l$)-Tripel. $l$ bezeichnet dabei das <b>Label</b> des Datenpunkts. Gehört der Datenpunkt zu der Klasse der Blaumeiseneier, so ist $l=0$, ansonsten ist $l=1$. Durchmische die Liste und unterteile die Daten in Trainings- und Testdaten in einem Verhältnis von 8:2.</i>
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
        ____


        <img style="float: left;" src="public/img/laptop_icon.png" width=50 height=50 /> <br><br>

        <i>Zähle wie viele Punkte das Perzeptron mit deiner Wahl der Gewichte $w_1$ und $w_2$ und des Bias $b$ richtig klassifiziert und gebe die Genauigkeit in Prozent aus. Benutze dabei die von dir implementierte Funktion aus der ersten Aufgabe.</i>
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
        In unserem Datensatz gibt es gleich viele Blaumeisen- und Greifvogel-Eier. Wenn also jemand rät und sich jeweils zufällig für eine Klasse entscheidet, liegt seine Rategenauigkeit bei 50%. Wenn dein Perzeptron eine ähnliche Genauigkeit aufweist, dann ist dieses Ergebnis nicht besonders erstrebenswert. Im nächsten Abschnitt schauen wir uns an, mit welchem Algorithmus die Gewichte angepasst werden, sodass das Perzeptron alle unsere Daten richtig klassifizieren kann.

        <h3>Geometrische Betrachtung</h3>

        Um die Funktionsweise des Perzeptrons anschaulich darzustellen, betrachten wir wie bisher unsere zweidimensionalen Datenpunkte. Das Perzeptron kann aber auch $n$-dimensionale Datenpunkte für beliebige ganzzahlige $n$ klassifizieren. Zur besseren Veranschaulichung nehmen wir in den folgenden Abschnitten außerdem an, dass $b = 0$ ist. Demzufolge ist der Output des Perzeptrons $f(w_1 \cdot x_1 + w_2 \cdot x_2)$. Bei der Aktivierungsfunktion $f$ handelt es sich um die bereits vorgestellte Treppenfunktion.

        ____

        <i style="font-size:38px">?</i>&nbsp;

        <i>Wir betrachten genau die Grenze der beiden Klassen, die zwar noch zur Klasse 0 gehört, aber trotzdem die Klassen voneinander trennt. Diese Grenze wird durch die Gleichung $w_1 \cdot x_1 + w_2 \cdot x_2 = 0$ beschrieben. An was erinnert dich das? <br> Tipp: Setze für $w_1$ und $w_2$ konktrete Werte ein, um dir den Sachverhalt besser zu veranschaulichen.</i>

        <details>
        <summary>➤ Klick hier, um deine Antwort zu prüfen.</summary>

        Die Gleichung beschreibt eine Gerade. Du kannst die Gleichung auch nach $x_2$ umstellen, sodass du die gewohnte Form einer Geradengleichung erhältst, die du aus der Schule kennst: $$x_2 = - \frac{w_1}{w_2} x_1 + 0.$$ <br><br>

        Wenn $w_1 \cdot x_1 + w_2 \cdot x_2 \leq 0$, dann wird dem Punkt $(x_1, x_2)$ die Ausgabe 0 zugewiesen. Wenn $w_1 \cdot x_1 + w_2 \cdot x_2 > 0$, dann die Ausgabe $1$. Durch die Gleichung $w_1 \cdot x_1 + w_2 \cdot x_2 = 0$ wird also eine Entscheidungsgrenze in Form einer Geraden beschrieben. Alle Punkte, die auf dieser Geraden liegen, gehören „gerade noch so” der Klasse $0$ an. Wenn ein Punkt auf der einen Seite der Geraden liegt, so wird ihm $0$ als Ausgabe zugewiesen und wenn ein Punkt auf der anderen Seite der Geraden liegt, so wird ihm $1$ zugewiesen. <br><br>
            
        Alternativ kannst du die „Entscheidungsgerade”, auch über den Gewichtsvektor $\vec{w} = \begin{pmatrix} w_1 \\ w2 \end{pmatrix}$ konstruieren. Die Gerade (blau dargestellt) läuft durch den Ursprung $(0,0)$ und ist orthogonal (rechtwinklig) zum Gewichtsvektor (schwarz dargestellt).
        <br><br>    
        <figure>
          <img src="public/img/vector.png" alt="weight vector" style="width:30%">
        </figure> 

            
        Zusatzfrage: Was ändert sich, wenn b $\neq$ 0?
            
        </details>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h3>Update der Gewichte</h3>

        Im letzten Abschnitt haben wir die geometrische Bedeutung der Gleichung $w_1 \cdot x_1 + w_2 \cdot x_2 = 0$ kennen gelernt. In diesem Abschnitt finden wir heraus, wie wir die Gewichte anpassen, wenn das Perzeptron Datenpunkte falsch klassifiziert.

        ____

        <i style="font-size:38px">?</i>&nbsp;
            
        <i>Der Punkt P $(-2, 0)$ wird vom Perzeptron falsch, der Punkt Q $(-1,1)$ wird dagegen richtig klassifiziert. Wie muss die Gerade (bzw. der <b>Gewichtsvektor</b>) angepasst werden, damit beide Punkte richtig klassifiziert werden? Beziehe in dein Update den Vektor $\vec{p} = \begin{pmatrix} -2 \\ 0 \end{pmatrix}$ mit ein.</i>
            
        &nbsp;


         <figure>
          <img src="public/img/vector2.png" alt="weight vector" style="width:30%">
          <figcaption></figcaption>
        </figure> 

        &nbsp;
            
            
        <details>

        <summary>➤ Klicke hier, um die Lösung anzuzeigen.</summary>
            
        Das Perzeptron klassifiziert die Punkte richtig, wenn die Entscheidungsgrenze zwischen den beiden Punkten $P$ und $Q$ verläuft. Einen passenden neuen Gewichtsvektor erhältst du beispielsweise über $\vec{w}_{\text{neu}} = \vec{w}_{\text{alt}} + 0.6 \cdot \vec{p}$.
            
        <br><br>


         <figure>
          <img src="public/img/vector3.png" alt="weight vector" style="width:30%">
          <figcaption></figcaption>
        </figure> 
            
        </details>




        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ____


        <img style="float: left;" src="public/img/laptop_icon.png" width=50 height=50 /> <br><br>

        <i>Setze nun folgende Formeln zur Gewichtsanpassung als Funktion in Code um.</i>

        $$w_i^{\text{neu}} = w_i^{\text{alt}} + 0.1 \cdot (o - l) \cdot x_i$$

        $$b^{\text{neu}} = b^{\text{alt}} + 0.1 \cdot (o - l)$$

        <i>Dabei wir mit $o$ der Output des Perzeptrons und mit $l$ das tatsächliche Label des Datenpunkts bezeichnet. Die <b>Lernrate</b> ist in der Formel auf 0.1 gesetzt. Sie gibt an, wie stark sich der Gewichtsvektor verändern soll.</i>
        """
    )
    return


@app.cell
def _():
    '''
    @param x1: x1-Koordinate des Datenpunkts
    @param x2: x2-Koordinate des Datenpunkts
    @param label: Label des Datenpunkts (Klasse 0 = Blaumeisen-Ei, Klasse 1 = Greifvogel-Ei)
    @param eta: Lernrate
    '''
    def train(x1, x2, label, eta=0.1):
       pass
    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Nun sind wir bereit, unser Perzeptron zu <b>trainieren</b>. d.h. die Gewichte schrittweise an unsere Datenpunkte anzupassen. Wir geben dem Perzeptron jeden Datenpunkt als Input und überprüfen, ob das Perzeptron den Datenpunkt richtig klassifiziert. Wenn das nicht der Fall ist, rufen wir unsere Funktion auf, die wir gerade implementiert haben, und passen die Gewichte an.
        ____

        <img style="float: left;" src="public/img/laptop_icon.png" width=50 height=50 /> <br><br>

        <i>Du hast oben bereits einen Datensatz erzeugt. Trainiere das Perzeptron solange, bis es jeden Punkt richtig klassifiziert (maximal aber drei Epochen lang). Zeichne nach jeder Änderung des Gewichtsvektors $\vec{w}$ ein Schaubild mit den Punkten und der Entscheidungsgerade ein. Färbe die Punkte oberhalb der Geraden mit der Farbe der Blaumeisen-Eier und die Punkte unterhalb der Geraden mit der Farbe der Greifvogel-Eier.</i>
        """
    )
    return


@app.cell
def _():
    _w1 = -1
    _w2 = 1
    _b = 1.5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ____

        <img style="float: left;" src="public/img/laptop_icon.png" width=50 height=50 /> <br><br>

        <i>Wenn du alles richtig gemacht hast, sollte der Algorithmus nach wenigen Schritten die Datenpunkte durch eine Gerade voneinander trennen. Führe deinen Code nun mit dem unteren Datensatz noch einmal aus. Beobachte, was passiert, und passe gegebenenfalls die Lernrate an.</i>
        """
    )
    return


@app.cell
def _(
    farbe_eier_blaumeise,
    farbe_eier_greifvogel,
    helligkeit_eier_blaumeise,
    hoehe_eier_blaumeise,
    plt,
):
    hoehe_eier_mini_greifvogel = [2.4, 2.8, 2.15, 2.4, 2.9, 2.5, 2.2, 2.5, 3, 2.7]
    helligkeit_eier_mini_greifvogel = [0.55, 0.6, 0.63, 0.6, 0.63, 0.5, 0.33, 0.45, 0.51, 0.55]
    plt.scatter(hoehe_eier_blaumeise, helligkeit_eier_blaumeise, color=farbe_eier_blaumeise)
    plt.scatter(hoehe_eier_mini_greifvogel, helligkeit_eier_mini_greifvogel, color=farbe_eier_greifvogel)
    _daten = []
    for i in range(10):
        _daten.append((hoehe_eier_blaumeise[i], helligkeit_eier_blaumeise[i], 0))
        _daten.append((hoehe_eier_mini_greifvogel[i], helligkeit_eier_mini_greifvogel[i], 1))
    return helligkeit_eier_mini_greifvogel, hoehe_eier_mini_greifvogel, i


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ____

        <i style="font-size:38px">?</i>

        <i>Das Perzeptron kann Datenpunkte zweier Klassen durch eine Gerade trennen. Führe das folgende Feld aus. Erkennst du ein Problem? Führe deinen implementierten Algorithmus auf diesem Datensatz aus. Was passiert?</i>
        """
    )
    return


@app.cell
def _(farbe_eier_blaumeise, farbe_eier_greifvogel, plt):
    _daten = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for datenpunkt in _daten:
        color = farbe_eier_blaumeise if datenpunkt[2] == 0 else farbe_eier_greifvogel
        plt.scatter(datenpunkt[0], datenpunkt[1], color=color)
    return color, datenpunkt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Grenzen des Perzeptrons 


        Die Möglichkeiten des Perzeptrons sind begrenzt. Vielleicht hast du schon im letzten Abschnitt mindestens einen Nachteil entdeckt. Fallen dir weitere ein? Hast du eine Idee, warum das Perzeptron nicht so mächtig ist?
        ____

        <i style="font-size:38px">?</i>

            
        <i>Überlege dir, in welchen Szenarien das Perzeptron an seine Grenzen stößt und klicke anschließend die Lösungen an.</i>

        &nbsp;

        <details>
            
        <summary>➤ 1. Nachteil</summary>
           
        Wie das letzte Beispiel zeigt, ist das Perzeptron nicht in der Lage, alle Daten richtig zu klassifizieren, wenn die Datenpunkte unterschiedlicher Klassen nicht durch eine Gerade getrennt werden können. 
            
        &nbsp;
           
        </details>

        <details>
            
        <summary>➤ 2. Nachteil</summary>
           
        Das Perzeptron kann nur zwei Klassen von Datenpunkten durch eine Entscheidungsgerade voneinander trennen. Wenn der Datensatz mehr als zwei Klassen beinhaltet (bspw. drei Arten von Vogeleiern), ist das Perzeptron nicht in der Lage alle Datenpunkte richtig zu klassifizieren, da es nur $0$ oder $1$ ausgeben kann.
            
        &nbsp;

           
        </details>


        <details>
            
        <summary>➤ 3. Nachteil</summary>
           
        Der Lernalgorithmus des Perzeptrons ist nicht optimal, d.h. es gibt Fälle bei denen er nicht funktoniert oder sich nur sehr langsam der Lösung nähert.
           
        </details>

        ### Zusammenfassung

        Bevor wir zur nächsten Einheit übergehen, fassen wir hier kurz unsere wichtigsten Erkenntnisse zusammen. 

        Aus Sicht des Computers können allerlei Daten als Punkte in einem zwei-, drei- oder $n$-dimensionalen Raum aufgefasst werden. Das Perzeptron zieht eine Gerade / Ebene durch den betrachteten Raum, um die Daten entweder der Klasse 0 oder 1 zuzuordnen. Wenn nicht alle Punkte richtig klassifiziert werden, wird die Gerade in jedem Schritt leicht angepasst. Dieser Algorithmus stößt allerdings schnell an seine Grenzen, wenn die Datenpunkte unterschiedlicher Klassen nicht durch eine Gerade getrennt werden können, es mehr als zwei Klassen gibt oder die Startgewichte ungünstig gewählt wurden.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

