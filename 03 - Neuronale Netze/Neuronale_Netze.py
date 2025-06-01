import marimo

__generated_with = "0.13.10"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Neuronale Netze
    
        In der letzten Einheit haben wir das Perzeptron kennen gelernt, das durch Fehler lernt und in bestimmten Szenarien Daten richtig klassifizieren kann. Der Klassifikationsalgorithmus des Perzeptrons stößt allerdings schnell an seine Grenzen. In dieser Einheit schauen wir uns an, wie wir das Perzeptron schrittweise verbessern können. Diese Verbesserungen führen uns zu neuronalen Netzen, die die rasante Entwicklung der KI der letzten Jahre entscheidend prägten.
    
        KI wird in den nächsten Jahren immer mehr Aufgaben übernehmen, die jetzt noch von Menschen ausgeführt werden. Gleichzeitig schafft KI auch neue Berufe und Perspektiven. Eine wichtige Herausforderung der Zukuft ist u.a. die Gestaltung einer sinnvollen Zusammenarbeit zwischen Mensch und KI. 
    
        &nbsp;
    
        <figure>
          <img src="public/resources/img/artificial-intelligence.jpg" alt="Deep Neural Network" style="width:50%">
            &nbsp;
          <figcaption><i>Dieses Bild wurde übrigens von einer KI erzeugt.</i></figcaption>
        </figure> 
    
        Zum Einstieg in diese Einheit rufen wir uns den Aufbau des Perzeptrons in Erinnerung. Das Perzeptron besteht aus einer festen Anzahl Inputs (abhängig von den Dimensionen der Punkte, die als Datengrundlage dienen), Gewichten mit denen die Eingaben mulipliziert und zusammen mit dem Bias addiert werden und einer Aktivierungsfunktion. Diesen Aufbau bezeichnen wir im Folgenden als <b>Neuron</b>.
    
    
        <figure>
          <img src="public/resources/img/perzeptron.png" alt="perzeptron" style="width:70%">
        </figure> 
    
    
        ## Aufbau neuronaler Netze
    
        Im Folgenden ändern wir das Perzeptron Schritt für Schritt ab, um dessen Defizite zu beheben.
    
        ### Mehr als zwei Klassen klassifizieren und Performance steigern
    
        Um die Performance unserer KI zu steigern, schalten wir mehrere Neuronen hinter- und nebeneinander. Die Ausgabe eines Neurons dient nun als Eingabe von nachfolgenden Neuronen. Sind Neuronen parallel in einer Ebene angeordnet, wird die Gesamtheit dieser Neuronen als <b>Layer</b> (bzw. Schicht) bezeichnet. Das gesamte Konstrukt mehreren Neuronenschichten bezeichnet man als <b>neuronales Netz</b>. Wenn es mehrere verdeckte Schichten gibt, bezeichnet man das Netz als <b>tiefes neuronales Netz</b> (deep neural network).
    
        <figure>
          <img src="public/resources/img/nn1.png" alt="perzeptron" style="width:60%">
        </figure> 
    
        Um nicht nur zwei Klassen von Datenpunkten klassifizieren zu können, wird die Ausgabe durch mehrere Neuronen erweitert. Die Nummer des Neurons, das den größten Wert in der Ausgabeschicht ausgibt, ist auch die Ausgabe des gesamten neuronalen Netzes. Wenn es also fünf Ausgabeneuronen gibt und das mittlere den größten Wert hat, dann weist das neuronale Netz den Datenpunkt der Klasse 2 zu (Outputs 0 bis 5). Bisher sind die Ausgaben der Neuronen allerdings entweder 0 oder 1, so dass es oft zu einem Gleichstand kommen kann. Nicht nur deswegen sollten wir die bisherige Aktivierungsfunktion durch eine geeignetere ersetzen.
    
        ### Neue Aktivierungsfunktion
    
        Das Perzeptron kann nur dann Datenpunkte verschiedener Klassen voneinander trennen, wenn die Datenpunkte der unterschiedlichen Klassen durch eine Gerade getrennt werden können. Das wird u.a. durch die Treppenfunktion verursacht, die wir als Aktivierungsfunktion verwenden. Außerdem gehen durch die Weiterleitung von entweder 0 oder 1 viele Informationen verloren, weil es keine Werte dazwischen gibt. Die <b>Sigmoidfunktion</b> $sig$ oder die <b>ReLU-Funktion</b> $relu$ sind in vielen Fällen besser als Aktivierungsfunktionen der Neuronen geeignet. Für unsere neuronalen Netze werden wir hauptsächlich die ReLU-Funktion verwenden.
    
        $$ sig(x) = \dfrac{e^x}{e^x + 1} $$
    
    
        $$ relu(x) = \left\{
        \begin{array}{ll}
        0, & x \leq 0 \\
        x, & \, \textrm{sonst} \\
        \end{array}
        \right. $$
    
        <figure>
          <img src="public/resources/img/sigmoid_and_relu.png" alt="Sigmoid and ReLU" style="width:50%">
        </figure> 
    
        ### Softmax
    
        Jetzt fehlt nur noch eine kleine Änderung, um ein herkömmliches neuronales Netz zu erhalten. Wie im vorletzten Abschnitt bereits umrissen, wird die Klassifikation des Datenpunkts jetzt nicht mehr durch eine 0- oder 1-Ausgabe des letzten Neurons ermittelt, sondern durch die Nummer des Neurons in der Ausgabeschicht, das die größte Ausgabe hat. Durch die neue ReLU-Aktivierungsfunktion erhalten wir in der letzten Ausgabeschicht nicht mehr 0- oder 1-Ausgaben, sondern Werte größer oder gleich 0. 
        Um als Ausgabe des neuronalen Netzes die Wahrscheinlichkeit zu erhalten, mit der ein Datenpunkt einer Klasse zugeordnet wird, wird eine am Ende eine zusätzliche Schicht mit einer speziellen Aktivierungsfunktion (Softmax-Funktion) eingefügt, deren Gewichte nicht trainiert werden.
    
        <figure>
          <img src="public/resources/img/nn2.png" alt="perzeptron" style="width:80%">
        </figure> 
    
        Jetzt sind wir bereit unser erstes neuronales Netz in Code umzusetzen. Damit wir nicht alles selbst implementieren müssen, verwenden wir die Bibliothek <i>PyTorch</i>, die von einem Facebook-Forschungsteam entwickelt wurde.
    
        ## PyTorch
    
        PyTorch bietet eine sehr einfache Weise, neuronale Netze zu konstruieren. Gehe das folgende Codefeld durch und führe es aus, um mit den Funktionsaufrufen vertraut zu werden. Wir konstruieren dabei das obige neuronale Netze mit vier Eingabe- und drei Ausgabeneuronen.
        """
    )
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import warnings
    warnings.filterwarnings('ignore')

    class Net(nn.Module):

        # Im Konstruktor werden die unterschiedlichen Schichten definiert
        def __init__(self, num_in, num_out):

            # Der Konstruktur der Elternklasse muss aufgerufen werden.
            super(Net, self).__init__()

            self.name_model = "Netzi"

            # Durch den folgenden Funktionsaufruf wird eine Schicht mit num_in eingehenden 
            # und 5 ausgehenden Verbindungen konstruiert.
            # Den Namen der Schichten kannst du selbst festlegen.
            # fc steht für fully connected.
            self.fc1 = nn.Linear(num_in, 5)
            # Achte darauf, dass die folgende Schicht die Anzahl der eingehenden Verbindungen
            # mit den ausgehenden Verbindungen der letzten Schicht übereinstimmt.
            self.fc2 = nn.Linear(5, 5)
            # Standardmäßig wird zu jedem Neuron ein Bias hinzugefügt. Durch den Parameter
            # 'bias' kann das deaktiviert werden.
            self.fc3 = nn.Linear(5, num_out, bias=False)

            # ReLU-Funktion
            self.relu=torch.nn.ReLU()
            # Softmax-Funktion
            self.softmax = torch.nn.Softmax()

        # In dieser Funktion muss festgelegt werden, wie die Eingabe durch das Netz propagiert wird (d.h. durch
        # die einzelnen Schichten „weitergereicht“ wird).
        def forward(self, x):
            # Zunächst wird die Eingabe mit den Gewichten der ersten Schicht multipliziert, 
            # in den einzelnen Neuronen aufsummiert und anschließend in die ReLU-Funktion eingesetzt.
            output = self.relu(self.fc1(x))
            # Die verarbeitete Eingabe wird nun durch die zweite Schicht propagiert. 
            output = self.relu(self.fc2(output))
            # In der vorletzten Schicht gibt es keine ReLU-Funktion mehr.
            output = self.fc3(output)
            # finale Ausgabe des neuronalen Netzes
            output = self.softmax(output)
            return output

    # Erzeugung eines Objekts des neuronalen Netzes
    erstes_nn = Net(4,3)
    print(f"Hallo mein Name ist {erstes_nn.name_model}!\n")
    print("Das ist mein Aufbau:\n")
    print(erstes_nn, "\n")
    print("Und das sind meine zufällig initialisierten Gewichte:")
    for param in erstes_nn.named_parameters():
        print("\n", param)

    # Das ist eine Testeingabe
    test_eingabe = torch.tensor([1.0, 2.5, -1, 0])
    # Die Ausgabe erhälst du entweder so
    print("\nAusgabe:", erstes_nn(test_eingabe))
    # oder durch den Funktionsaufruf forward(eingabe)
    print("Ausgabe:", erstes_nn.forward(test_eingabe))
    return erstes_nn, nn, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        ____
    
        <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>
    
        <i>Im letzten Codefeld wurden unser erstes neuronales Netz erzeugt. Lies die gesuchten Gewichte anhand der letzten Ausgabe ab und überprüfe deine Eingabe, indem du das Codefeld ausführst. Runde gegebenenfalls die Eingaben auf die vierte Nachkommastelle ab.</i>
        """
    )
    return


@app.cell
def _(erstes_nn):
    from public.resources.code.help_functions import pruefe_gewichte

    # Ersetze die Nullen durch die richtigen Werte.

    # Gewicht zwischen dem ersten Neuron der Eingabeschicht und dem ersten Neuron der ersten verdeckten Schicht
    gewicht1 = 0

    # Bias des letzen Neurons der ersten verdeckten Schicht
    gewicht2 = 0

    # Bias des zweiten Neurons der zweiten verdeckten Schicht
    gewicht3 = 0

    # Gewicht zwischen dem vierten Neuron der zweiten verdeckten Schicht 
    # und dem dritten Neuron der dritten verdeckten Schicht
    gewicht4 = 0

    print(pruefe_gewichte(erstes_nn, gewicht1, gewicht2, gewicht3, gewicht4))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ____
    
        <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>
    
        <i>Jetzt bist du bereit ein neuronales Netz eigenständig zu konstruieren. Implementiere das abgebildete neuronale Netz und gib das Ergebnis des durchpropagierten Datenpunkts an.</i>
    
        <figure>
          <img src="public/resources/img/nn3.png" alt="neuronales Netz" style="width:60%">
          <figcaption></figcaption>
        </figure>
        """
    )
    return


@app.cell
def _(torch):
    datenpunkt = torch.tensor([1.0, 2.0])

    # Füge hier deinen Code ein.
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Bis jetzt haben wir zwar neuronale Netze konstruiert, aber sie noch nicht trainieren lassen. Die vorhandenen Trainingsdaten müssen wir nutzen, um die Gewichte so anzupassen, dass das neuronale Netz auf den Testdaten (die wir nicht für das Training benutzen) gute Ergebnisse erzielt. Im nächsten Abschnitt schauen wir uns an, wie das funktioniert.
    
        ## Backpropagation
    
        Der Algorithmus, der die Gewichte der neuronalen Netze abändert und ein entscheidender Faktor am Erfolg von Deep-Learning-Algorithmen ist, ist der <b>Backpropagation-Algorithmus</b>. Der Backpropagation-Algorithmus ist ein Optimierungsalgorithmus, d.h. bei der Funktion, die den Fehler des neuronalen Netzes beschreibt, wird (in diesem Fall) nach dem Minimum gesucht, weil wir den Fehler so klein wie möglich halten möchten. 
    
        Die Suche nach dem Minimum können wir uns mit folgendem Bild veranschaulichen. Ein Weihnachtsmann sitzt in seinem E-Schlitten auf einem Hügel und möchte den Weg ins Tal finden. Leider kennt er den Weg dahin nicht. Zu allem Überfluss ist es auch schon dunkel und sogar etwas nebelig ist, sodass er nur zehn Meter weit sehen kann. Er kann aber um sich herum erkennen, in welche Richtung der Hügel am steilsten abfällt. (In diese Richtung zeigt übrigens auch die Ableitung der Funktion, die das Gelände beschreibt.) Er stellt sein E-Schlitten so ein, dass er eine bestimmte Distanz in die Richtung des steilsten Abstiegs fährt, anschließend stoppt, die Richtung des Abstiegs noch einmal neu bestimmt und in diese Richtung wieder eine bestimmte Distanz fährt. Wenn alles optimal verläuft, findet er auf diese Weise den Weg ins Tal.
    
        <figure>
          <img src="public/resources/img/loss_function.png" alt="Verlustfunktion" style="width:60%">
          <figcaption></figcaption>
        </figure> 
    
        Analog dazu funktioniert auch die Optimierung bei neuronalen Netzen. Die Funktion, deren globales Minimum erreicht werden soll, heißt <b>Verlustfunktion / Loss-Funktion </b>. Die Funktion MSE (Mean Squared Error) ist ein Beispiel für so eine Funktion:
    
        $$MSE = \dfrac{1}{n} \bigl[ (y_1 - o_1)^2 + (y_2 - o_2)^2 + \dots + (y_n - o_n)^2 \bigr], $$
    
        wobei $(y_1, \dots, y_n)$ die optimale und $(o_1, \dots, o_n)$ die tatsächliche Ausgabe eines neuronalen Netzes beschreibt. 
    
        ____
    
        <i style="font-size:38px">?</i>
    
    
        <i>Wenn wir z.B. einen Datenpunkt betrachten, der ein Blaumeisen-Ei repräsentiert, dann ist die optimale Ausgabe bei drei möglichen Klassen (Klasse 0 = Blaumeisen, Klasse 1 = Ente, Klasse 2 = Greifvogel) der Vektor $(1, 0, 0)$. Wenn die tatsächliche Ausgabe des neuronalen Netzes $(0.5, 0.25, 0.25)$ ist, was ist dann der Verlust nach der oberen Formel?</i>
    
        <details>
    
        <summary>➤ Klicke hier, um deine Antwort zu prüfen.</summary>
    
        $$\dfrac{1}{3} \bigl[ (1 - 0.5)^2 + (0 - 0.25)^2 + (0 - 0.25)^2 \bigr] = 0.375.$$
    
        Wenn das neuronale Netz nur ein Gewicht hat, könnte die Verlustfunktion so aussehen:
    
        <figure>
          <img src="public/resources/img/loss_function2.png" alt="Verlustfunktion" style="width:45%">
        </figure> 
    
        Das aktuelle Gewicht $w_1$ von $0.7$ muss also ein bisschen vergrößert werden, um den Verlust zu verkleinern.
    
        </details>
    
        Wenn das neuronale Netz nur zwei Gewichte hat, könnte eine Verlustfunktion wie folgt aussehen. Bei mehr als zwei Gewichten (in der Praxis eingesetzte neuronale Netze haben Millionen von trainierbaren Gewichten) ist eine Visualisierung allerdings nicht mehr so einfach möglich.
    
        <figure>
          <img src="public/resources/img/train_val_loss_landscape.png" alt="Loss-Function" style="width:50%">
        </figure> 
    
        Wenn wir bestimmt haben, ob wir ein Gewicht verkleinern oder vergrößern müssen, um den Verlust zu reduzieren, müssen wir noch festlegen, wie stark wir das Gewicht verändern möchten. Dabei können unterschiedliche Probleme auftreten. Ist die Veränderung des Gewichts zu gering, kann es sein, dass das neuronale Netz in einem lokalen Minimum stecken bleibt oder sich nur sehr langsam dem globalen Minimum nähert. Verändern wir das Gewicht zu stark, ist es möglich, dass wir über das Ziel hinausschießen. 
    
        <figure>
          <img src="public/resources/img/loss_function3.png" alt="Verlustfunktion" style="width:95%">
        </figure> 
    
        Wir müssen also die <b>Lernrate</b> des neuronalen Netzes mit Bedacht wählen und möglicherweise immer wieder anpassen. Die Update-Regel für jedes Gewicht $w$ im neuronalen Netz können wir folgendermaßen notieren:
    
        $$w_{\text{neu}} \longleftarrow w_{\text{alt}} - \alpha \cdot \Delta w.$$
    
        $\alpha$ ist die Lernrate und $\Delta w$ der Gradient (die Ableitung) des Gewichts. Der Gradient gibt nicht nur die Richtung an, in der das Gewicht verändert werden muss, sondern beschreibt auch, wie stark das betrachtete Gewicht zu dem Verlust beigetragen hat. 
    
        Den Gradienten eines Gewichts $w$ bestimmen wir, indem wir die Verlustfunktion nach $w$ durch mehrfache Anwendung der Kettenregel ableiten. Da dieser Prozess sehr mühselig ist, verzichten wir an dieser Stelle auf weitere Details, weil PyTorch für uns diese Arbeit übernehmen wird.
    
        <figure>
          <img src="public/resources/img/backpropagation.png" alt="Verlustfunktion" style="width:65%">
        </figure> 
    
        Die Berechnung der Gradienten bei der Backpropagation erfordert sehr viel Rechenaufwand. Eine CPU wird nur bei kleinen Daten(mengen) gute Ergebnisse in überschaubarer Zeit liefern können. Aus diesem Grund verwendet man GPU-Einheiten (Grafikprozessoren), um ein neuronales Netz trainieren zu lassen. Der Vorteil dieser Verwendung besteht darin, dass die Berechnungen <i>parallel</i> ablaufen können und das Netz somit viel schneller trainiert.
    
        ## Training eines neuronalen Netzes
    
        Nach so viel Theorie können wir endlich neuronale Netze trainieren lassen! Untersuche den Code, um dein eigenes neuronales Netz weiter unten an die Daten anzupassen.
        """
    )
    return


@app.cell
def _(torch):
    from public.resources.code.help_functions import daten, datenpunkte_zeichnen
    import matplotlib.pyplot as plt
    from matplotlib import colors
    torch.manual_seed(1)
    (x_train, y_train, x_test, y_test) = daten()
    print(f'Wir haben {len(y_train)} Trainingsdatenpunkte und {len(y_test)} Testdatenpunkte zur Verfügung.')
    datenpunkte_zeichnen(x_train, y_train, ['#ec90cc', '#4f7087'])
    return datenpunkte_zeichnen, x_test, x_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
        Wir implementieren folgendes neuronales Netz, das du bereits oben konstruiert hast.
    
        &nbsp;
    
    
         <figure>
          <img src="public/resources/img/nn3.png" alt="neuronales Netz" style="width:60%">
          <figcaption></figcaption>
        </figure> 
    
        &nbsp;
        """
    )
    return


@app.cell
def _(nn, torch):
    class Net_1(nn.Module):

        def __init__(self, num_in, num_out):
            super(Net_1, self).__init__()
            self.fc1 = nn.Linear(num_in, 2)
            self.fc2 = nn.Linear(2, 2)
            self.relu = torch.nn.ReLU()
            self.softmax = torch.nn.Softmax()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.softmax(self.fc2(x))
            return x
    return (Net_1,)


@app.cell
def _(torch):
    # Mit dieser Funktion wird die Genauigkeit von 
    # einem neuronalen Netz auf einem Datensatz gemessen.

    '''
    @param model: neuronales Netz auf dem die Messung durchgeführt wird
    @param x: Datenpunkte 
    @param labels: Labels zu den Datenpunkten
    @param name: Name des Datensatzes z.B. Training oder Test
    '''
    def evaluation(model, x, labels):
        model.train(False)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(outputs, labels)
        correct = sum(torch.eq(preds, labels)).item()
        total = len(labels)
        accuracy = round(correct/total, 3) * 100
        return round(loss.item(), 5), accuracy
    return (evaluation,)


@app.cell
def _(Net_1, torch):
    net = Net_1(2, 2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.CrossEntropyLoss()
    epochs = 100
    return epochs, loss_func, net, optimizer


@app.cell
def _(
    epochs,
    evaluation,
    loss_func,
    net,
    optimizer,
    x_test,
    x_train,
    y_test,
    y_train,
):
    # Hier evaluieren wir die Genauigkeit des untrainierten neuronalen Netzes.
    loss_res, accuracy_res = evaluation(net, x_test, y_test)
    print(f"Untrainiertes neuronales Netz - Ergebniss für Trainingsdatensatz: Genauigkeit={accuracy_res}%, Loss={loss_res}")

    for e in range(epochs):

        ''' Trainingsprozess '''

        # Das neuronale Netz wird in den Trainingsmodus versetzt (d.h. es werden Gradienten berechnet).
        net.train(True)

        # Alle Trainingspunkte werden durch das neuronale Netz propagiert. 
        outputs = net(x_train)

        # Der Loss hängt von der Ausgabe des neuronalen Netzes und den tatsächlichen Labeln ab.
        loss = loss_func(outputs, y_train)

        # Alle berechneten Gradienten vom letzten Durchgang werden gelöscht.
        optimizer.zero_grad()

        # Mit diesem Funktionsaufruf werden die Gradienten berechnet.
        loss.backward()

        # Hier wird für jedes Gewicht ein Update durchgeführt.
        optimizer.step()

        ''' Evaluation auf den Trainings- und Testdaten '''

        # Evaluation wird für jede 5. Epoche durchgeführt. 
        if e % 5 == 0:
            loss_res, accuracy_res = evaluation(net, x_train, y_train)
            print(f"{e}. Epoche - Ergebniss für Trainingsdatensatz: Genauigkeit={accuracy_res}%, Loss={loss_res}")
            loss_res, accuracy_res = evaluation(net, x_test, y_test)
            print(f"{e}. Epoche - Ergebniss für Testdatensatz: Genauigkeit={accuracy_res}%, Loss={loss_res}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ____
    
        <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>
    
        <i>Du kennst nun alle Codebausteine, um dein eigenes neuronales Netz zu konstruieren und es trainieren zu lassen. Setze ein neuronales Netz für die folgenden Daten um und passe die Gewichte an den Datensatz an. Brich das Training ab, sobald das Netz eine 93%-Genauigkeit auf dem Trainingsdatensatz erzielt. Speichere außerdem in jeder Epoche das Netz, das über alle vergangenen Durchläufe hinweg die höchste Genauigkeit erreicht hat.</i>
        """
    )
    return


@app.cell
def _(datenpunkte_zeichnen):
    from public.resources.code.help_functions import daten2
    (x_train_1, y_train_1, x_test_1, y_test_1) = daten2()
    print(f'Wir haben {len(y_train_1)} Trainingsdatenpunkte und {len(y_test_1)} Testdatenpunkte zur Verfügung.')
    datenpunkte_zeichnen(x_train_1, y_train_1, ['#ec90cc', '#8b4513', '#4f7087'])
    return


@app.cell
def _():
    # Implementiere hier die Klasse für dein neuronales Netz.
    return


@app.cell
def _():
    # Erzeuge hier das Objekt deiner Klasse und den Optimizer. 
    # Lege hier außerdem deine Loss-Funktion und die Anzahl der Epochen fest.
    return


@app.cell
def _():
    # Implementiere hier deinen Trainingsprozess.
    # Breche den Trainingsprozess ab, wenn eine Genauigkeit von 93% auf den
    # Trainingsdaten erreicht wurde.
    # Speicher außerdem immer das bisher beste Model mit deepcopy(model) 
    # in einer Variablen ab.

    from copy import deepcopy
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <h2>Bildquellen</h2>
    
        https://pixabay.com/de/photos/ai-generiert-junge-junger-mann-7772478/
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
