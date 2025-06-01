import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell(hide_code=True)
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

    Jetzt sind wir bereit unser erstes neuronales Netz in Code umzusetzen. Damit wir nicht alles selbst implementieren müssen, verwenden wir die Bibliothek <i>Numpy</i>.

    ## Numpy

    Numpy bietet eine sehr einfache Weise, neuronale Netze zu konstruieren. Gehe das folgende Codefeld durch und führe es aus, um mit den Funktionsaufrufen vertraut zu werden. Wir konstruieren dabei das obige neuronale Netze mit vier Eingabe- und drei Ausgabeneuronen.
    """
    )
    return


@app.cell
def _(FullyConnectedLayer, relu, softmax):
    import numpy as np

    class Net:
        def __init__(self, num_in, num_out):
            self.name_model = "Netzi"

            # Define layers that mimic the torch layers
            self.fc1 = FullyConnectedLayer(num_in, 5)      # first layer: num_in -> 5 
            self.fc2 = FullyConnectedLayer(5, 5)             # second layer: 5 ->5
            self.fc3 = FullyConnectedLayer(5, num_out, bias=False)  # third layer: 5 -> num_out, no bias

        def forward(self, x):
            # First layer + ReLU
            x = relu(self.fc1.forward(x))
            # Second layer + ReLU
            x = relu(self.fc2.forward(x))
            # Third layer (no ReLU here)
            x = self.fc3.forward(x)
            # Softmax on output layer
            x = softmax(x)
            return x

        def __str__(self):
            # Simple representation of the network layers
            rep = f"Model name: {self.name_model}\n"
            rep += "Layer 1 (fc1):\n" + str(self.fc1) + "\n"
            rep += "Layer 2 (fc2):\n" + str(self.fc2) + "\n"
            rep += "Layer 3 (fc3):\n" + str(self.fc3)
            return rep

    # Create an instance of the network with 4 input features and 3 output features.
    erstes_nn = Net(4, 3)
    print(f"Hallo mein Name ist {erstes_nn.name_model}!\n")

    print("Das ist mein Aufbau:\n")
    print(erstes_nn, "\n")

    print("Und das sind meine zufällig initialisiert/en Gewichte:")
    # Display weights and biases of each layer
    print("\nfc1 layer parameters:\n", "Weights:\n", erstes_nn.fc1.weight)
    if erstes_nn.fc1.use_bias:
        print("Bias:\n", erstes_nn.fc1.bias)

    print("\nfc2 layer parameters:\n", "Weights:\n", erstes_nn.fc2.weight)
    if erstes_nn.fc2.use_bias:
        print("Bias:\n", erstes_nn.fc2.bias)

    print("\nfc3 layer parameters:\n", "Weights:\n", erstes_nn.fc3.weight)
    if erstes_nn.fc3.use_bias:
        print("Bias:\n", erstes_nn.fc3.bias)

    # Test input vector
    test_eingabe = np.array([1.0, 2.5, -1, 0])

    # Make predictions using the forward method.
    ausgabe = erstes_nn.forward(test_eingabe)
    print("\nAusgabe (Ergebnis des Vorwärtsdurchlaufs):", ausgabe)
    return erstes_nn, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Aufgabe 1""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Aufgabe 2""")
    return


@app.cell(hide_code=True)
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
def _(np):
    datenpunkt = np.array([1.0, 2.0])

    # Füge hier deinen Code ein.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Tipp 1": "Sieh dir nochmal die Definition von `class Net` an.",
        "Tipp 2": "Du brauchst um ein eigenes neuronales Netz zu implementieren eine eigene Klasse mit 2 Eingaben und 2 Ausgaben sowie 2 versteckten Schichten",
        "Tipp 3": "Tipp 3",
        "Lösung":mo.md(
        r"""
    ```python
    class Net_1:
            def __init__(self, num_in, num_out):
                # First fully connected layer maps from num_in to 2
                self.fc1 = FullyConnectedLayer(num_in, 2)
                # Second fully connected layer maps from 2 to 2
                self.fc2 = FullyConnectedLayer(2, 2)

            def forward(self, x):
                x = relu(self.fc1.forward(x))
                x = self.fc2.forward(x)
                x = softmax(x)
                return x

            def __str__(self):
                rep = "Net_1 architecture:\n"
                rep += "Layer 1 (fc1):\n" + str(self.fc1) + "\n"
                rep += "Layer 2 (fc2):\n" + str(self.fc2) + "\n"
                return rep
    ```
    """
    ),
    })
    return


@app.cell(hide_code=True)
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
def _():
    from public.resources.code.help_functions import daten, datenpunkte_zeichnen
    import matplotlib.pyplot as plt
    from matplotlib import colors

    (x_train, y_train, x_test, y_test) = daten()
    print(f'Wir haben {len(y_train)} Trainingsdatenpunkte und {len(y_test)} Testdatenpunkte zur Verfügung.')
    datenpunkte_zeichnen(x_train, y_train, ['#ec90cc', '#4f7087'])
    return datenpunkte_zeichnen, x_train, y_train


@app.cell(hide_code=True)
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
def _(FullyConnectedLayer, np):
    class Net_1:
        def __init__(self, num_in, num_out):
            # First fully connected layer maps from num_in to 2
            self.fc1 = FullyConnectedLayer(num_in, 2)
            # Second fully connected layer maps from 2 to 2
            self.fc2 = FullyConnectedLayer(2, 2)

        def forward(self, x):
            self.z1 = self.fc1.forward(x)
            self.a1 = np.maximum(0, self.z1)  # ReLU activation
            self.z2 = self.fc2.forward(self.a1)
            # Numerically stable softmax:
            exp_scores = np.exp(self.z2 - np.max(self.z2))
            self.out = exp_scores / np.sum(exp_scores)
            return self.out

        # For training purposes, you’d also implement a function to compute gradients.
        # This example provides a skeleton for backpropagation:
        def backward(self, x, label):
            # Assume label is an integer representing the true class.
            # Compute gradient of the loss (cross-entropy) wrt. the output (softmax output).
            y_true = np.zeros_like(self.out)
            y_true[label] = 1

            # dL/dz2 for softmax + cross-entropy simplifies to (y_pred - y_true)
            delta2 = self.out - y_true  

            # Gradients for fc2 parameters:
            self.fc2.grad_weights = np.outer(delta2, self.a1)
            if self.fc2.use_bias:
                self.fc2.grad_bias = delta2

            # Backprop through fc2 to hidden layer:
            delta1 = np.dot(self.fc2.weight.T, delta2)
            # Backprop through ReLU:
            delta1[self.z1 <= 0] = 0

            # Gradients for fc1 parameters:
            self.fc1.grad_weights = np.outer(delta1, x)
            if self.fc1.use_bias:
                self.fc1.grad_bias = delta1

        def update_params(self, lr=0.1):
            # Update fc1 weights and bias:
            self.fc1.weight -= lr * self.fc1.grad_weights
            if self.fc1.use_bias:
                self.fc1.bias -= lr * self.fc1.grad_bias
            # Update fc2 weights and bias:
            self.fc2.weight -= lr * self.fc2.grad_weights
            if self.fc2.use_bias:
                self.fc2.bias -= lr * self.fc2.grad_bias


        def __str__(self):
            rep = "Net_1 architecture:\n"
            rep += "Layer 1 (fc1):\n" + str(self.fc1) + "\n"
            rep += "Layer 2 (fc2):\n" + str(self.fc2) + "\n"
            return rep
    return (Net_1,)


@app.cell
def _(np):
    def cross_entropy_loss(outputs, labels, epsilon=1e-10):
        """
        Compute the average cross entropy loss.

        Parameters:
            outputs: An array of shape (N, num_classes) with the predicted probabilities.
            labels:  A 1D array of length N with the true labels (integer class indices).
            epsilon: Small constant to avoid log(0).

        Returns:
            The average cross entropy loss.
        """
        # Select the probabilities corresponding to the true labels
        probs = outputs[np.arange(len(labels)), labels]
        # Compute the negative log likelihood for each sample
        losses = -np.log(probs + epsilon)
        # Return the mean loss
        return np.mean(losses)

    def evaluation(model, x, labels):
        """
        Evaluate a neural network model on a given dataset.

        Parameters:
          model:     A neural network model with a `forward` method.
          x:         Input data. Expected to be a 2D numpy array of shape (N, features).
                     Alternatively, if x is a list of 1D arrays, each representing a sample.
          labels:    True labels as a 1D numpy array with integer class indices.

        Returns:
          A tuple (loss, accuracy) where:
          - loss: average cross entropy loss over all samples (rounded to 5 decimals)
          - accuracy: classification accuracy in percent (rounded to three decimals)
        """
        # Assume model is in evaluation mode:
        # Note: In our NumPy-only model, there's no train()/eval() mode.

        outputs = []  # Will hold the output probability vector for each sample

        # If x is two-dimensional (N, features) then iterate over each sample.
        for sample in x:
            out = model.forward(sample)
            outputs.append(out)

        # Stack the outputs into a (N, num_classes) numpy array.
        outputs = np.vstack(outputs)

        # Get the predicted labels using argmax along axis=1.
        preds = np.argmax(outputs, axis=1)

        # Compute cross entropy loss using our helper function.
        loss_value = cross_entropy_loss(outputs, labels)

        # Compute accuracy in percentage
        correct = np.sum(preds == labels)
        total = labels.shape[0]
        accuracy = round(correct / total, 3) * 100

        return round(loss_value, 5), accuracy
    return cross_entropy_loss, evaluation


@app.cell
def _(Net_1, evaluation, np):
    net = Net_1(2,2)

    # Create a dummy dataset: 10 samples with 4 features each.
    x_data = np.random.randn(10, 2)
    # Random integer labels from 0 to 2.
    labels = np.random.randint(0, 2, size=10)

    # Evaluate the dummy model.
    loss1, accuracy = evaluation(net, x_data, labels)
    print("Loss:", loss1)
    print("Accuracy:", accuracy)
    return (net,)


@app.cell
def _():
    lr = 0.1
    epochs = 100
    return epochs, lr


@app.cell
def _(cross_entropy_loss, epochs, lr, net, np, x_train, y_train):
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0

        # In this simple example, we loop over only one sample.
        for x, label in zip([x_train], y_train):
            # Forward pass:
            output = net.forward(x[0])
            # Compute loss:
            loss = cross_entropy_loss(output, label)
            epoch_loss += loss

            # Prediction:
            pred = np.argmax(output)
            if pred == label:
                correct += 1

            # Backward pass:
            net.backward(x, label)
            # Update parameters (this is like optimizer.step() in PyTorch)
            net.update_params(lr)

        accuracy1 = correct / 1 * 100  # For one sample, accuracy is either 0 or 100%
        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.5f} | Accuracy: {accuracy1}%")

    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <h2>Bildquellen</h2>

    https://pixabay.com/de/photos/ai-generiert-junge-junger-mann-7772478/
    """
    )
    return


@app.cell
def _(np):
    np.random.seed(0)

    class FullyConnectedLayer:
        def __init__(self, input_dim, output_dim, bias=True):
            self.out = output_dim
            # Initialize weights with a small random numbers.
            self.weight = np.random.randn(output_dim, input_dim) * 0.1
            self.use_bias = bias
            if self.use_bias:
                self.bias = np.random.randn(output_dim) * 0.1
            else:
                self.bias = None
            # Placeholders for gradients
            self.grad_weights = np.zeros_like(self.weights)
            if self.use_bias:
                self.grad_bias = np.zeros_like(self.bias)

        def forward(self, x):
            # x is assumed to be a 1D input vector of shape (input_dim,)
            # Compute linear transformation y = W*x + b
            y = np.dot(self.weight, x)
            if self.use_bias:
                y += self.bias
            return y


        # For training purposes, you’d also implement a function to compute gradients.
        # This example provides a skeleton for backpropagation:
        def backward(self, x, label):
            # Assume label is an integer representing the true class.
            # Compute gradient of the loss (cross-entropy) wrt. the output (softmax output).
            y_true = np.zeros_like(self.out)
            y_true[label] = 1

            # dL/dz2 for softmax + cross-entropy simplifies to (y_pred - y_true)
            delta2 = self.out - y_true  

            # Gradients for fc2 parameters:
            self.fc2.grad_weights = np.outer(delta2, self.a1)
            if self.fc2.use_bias:
                self.fc2.grad_bias = delta2

            # Backprop through fc2 to hidden layer:
            delta1 = np.dot(self.fc2.weight.T, delta2)
            # Backprop through ReLU:
            delta1[self.z1 <= 0] = 0

            # Gradients for fc1 parameters:
            self.fc1.grad_weights = np.outer(delta1, x)
            if self.fc1.use_bias:
                self.fc1.grad_bias = delta1

        def update_params(self, lr=0.1):
            # Update fc1 weights and bias:
            self.fc1.weight -= lr * self.fc1.grad_weights
            if self.fc1.use_bias:
                self.fc1.bias -= lr * self.fc1.grad_bias
            # Update fc2 weights and bias:
            self.fc2.weight -= lr * self.fc2.grad_weights
            if self.fc2.use_bias:
                self.fc2.bias -= lr * self.fc2.grad_bias


        def __str__(self):
            s = f'Weights shape: {self.weight.shape}\n'
            if self.use_bias:
                s += f'Bias shape: {self.bias.shape}\n'
            else:
                s += 'Bias disabled\n'
            return s

    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        # Numerically stable softmax
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


    return FullyConnectedLayer, relu, softmax


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
