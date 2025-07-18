import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Bilderklassifikation

    In der letzten Einheit haben wir gelernt, wie wir in PyTorch neuronale Netze implementieren und trainieren können. Als Datengrundlage dienten dabei zweidimensionale Punkte, die Vogeleier repräsentierten. Jeder dieser Dimensionen stand für eine Eigenschaft des Eis. So beschrieb der Datenpunkt $(8, 0.1)$ ein $8$ cm hohes und relativ dunkles Ei. Diese Klassifizierung setzt voraus, dass wir 

    <ul>
        <li>festlegen, welche Eigenschaften der zu klassifizierenden Objekte für eine Unterscheidungen zur anderen Art der Objekte relevant sind und</li>
        <li>die festgelegten Eigenschaften für jedes Obejekt ausmessen.</li>
    </ul>

    Die Umsetzung beider Aspekte ist sehr schwierig und kostspielig. Es wäre viel praktischer, wenn wir Bilder von den verschiedenen Vogeleiern der KI zur Verfügung stellen würden und die KI die relevanten Eigenschaften der Objekte selbst herausfinden und ausmessen könnte. Genau das möchten wir in dieser Einheit realisieren. 

    ## Codierung von Bildern

    Um es so einfach wie möglich zu halten, betrachten wir im Folgenden nur Graustufenbilder. Bei Graustufenbildern wird ein Zahlenwert pro Pixel gespeichert. Der Zahlenwert 0 entspricht einem komplett schwarzen Pixel und der Wert 255 einem weißen Pixel. Zahlenwerte zwischen 0 und 255 entsprechen unterschiedlichen Graustufen. 

    <figure>
      <img src="public/resources/img/lincoln_pixels.png" alt="Abraham Lincoln Pixels" style="width:50%">
    </figure> 

    Bilder sind also nichts anderes als zusammgesetzte Pixel und für den Computer somit einfach nur Listen aus Zahlen, die sich als Eingaben für neuronale Netze sehr gut eignen. Die Anzahl der Pixel muss dabei der Anzahl der Neuronen der Eingabeschicht entsprechen. Ein Bild, das nur aus einem Pixel besteht, können wir als Punkt in einem eindimensionalen Koordinatensystem auffassen. Ein Bild aus zwei Pixeln ist ein Punkt im zweidimensionalen Koordinatensytem usw. Reale Bilder können demzufolge als $n$-dimensionale Punkte aufgefasst werden, die wir uns allerdings nicht mehr wirklich vorstellen können.


    <figure>
      <img src="public/resources/img/nn_img.png" alt="Bild in neuronales Netz" style="width:70%">
    </figure> 

    Wie auch Perzeptronen, trennen neuronale Netze Datenpunkte durch Grenzen voneinander, die sie durch die Trainingsdaten selbst erlernen. Diese Grenzen sind keine Geraden oder Ebenen wie beim Perzeptron, sondern gekrümmte $n$-dimensionale Objekte. Mit Hilfe der Trainingsdaten lernen neuronale Netze den (vermeintlichen) Verlauf dieser Grenzen, sodass sie anschließend ungesehene Daten den unterschiedlichen Klassen zuordnen können. In der unteren Abbildung ist zu sehen, wie ein neuronales Netz Trainingspunkte klassifizieren könnte. Den roten unbekannten Datenpunkt würde das neuronale Netz der Klasse 3 zuweisen.

    <figure>
      <img src="public/resources/img/nn_klassen.png" alt="Klassen" style="width:50%">
    </figure>

    Jetzt können wir also mit neuronalen Netzen auch Bilder klassifizieren, oder? Leider haben wir noch ein Problem... 

    <figure>
      <img src="public/resources/img/frosch_kermit.png" alt="Kermit the Frog" style="width:50%">
    </figure> 

    Fully-connected neuronale Netze sind nämlich ziemlich schlecht darin, Eigenschaften (<b>Feature</b>) von Objekten zu extrahieren bzw. zu erkennen. Wir müssen dieses neuronalen Netz also etwas abändern, um auch Bilder klassifizieren zu können.

    ## Convolutional Neural Networks (CNNs)

    <b>Convolutional Neural Networks (CNNs)</b> sind neuronale Netze, die Convolutional Layer enthalten. Convolutional Layer kann man sich als Filter vorstellen.

    Das Bild wird (zu Beginn) durch verschiedene Filtern gereicht. Ein Filter ist dabei nichts anderes als eine Zahlenmatrix, die durch das Bild geschoben wird. Durch diese Filter kann das CNN z.B. horizontale oder vertikale Kante erkennen. 

    <figure>
      <img src="public/resources/img/convolution.png" alt="Convolution" style="width:50%">
    </figure> 

    Die Gewichte von den Filtern erlernt das neuronale Netz dabei selbst. Anschließend wird ein sogenanntes Max-Pooling durchgeführt, d.h. aus einem bestimmten Bereich der Feature Maps wird die größte Zahl ausgewählt, sodass die räumlichen Bildinformationen auf einen kleineren Bereich heruntergebrochen werden. 

    <figure>
      <img src="public/resources/img/max_pooling.png" alt="Max-Pooling" style="width:50%">
    </figure> 

    Nach diesen (mehrmals durchgeführten) Convolutions wird die Eingabe am Ende in eine fully-connected Layer weitergereicht.

    <figure>
      <img src="public/resources/img/cnn_architecture.png" alt="Max-Pooling" style="width:90%">
    </figure>


    ## Vogeleierklassifikation

    Jetzt sind wir bereit ein eigenes CNN zu implementieren, das Bilder klassifizieren kann. Bei den Bildern handelt es sich um Vogeleier, die du und deine Mitschüler:innen gemalt haben. Dein neuronales Netz wird nach dem Training in der Lage sein, Blaumeisen-, Enten- und Greifvogel-Eier auseinanderhalten zu können. 

    <figure>
      <img src="public/resources/img/vogeleier.jpg" alt="Vogeleier" style="width:70%">
    </figure> 

    ____

    <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>

    <i>Ergänze die folgenden Codefelder den Kommentaren entsprechend, um dein eigenes neuronales Netz zu konstruieren, das die Bilder der Vogeleier richtig klassifizieren kann.</i>
    """
    )
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore')
    from public.resources.code.help_functions import ei_zeichnen
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn


    # Füge hier den relativen Pfad zu deinen Trainings- und Testdaten 
    # ausgehend von dieser Datei ein.

    TRAIN_DATA_PATH = '04 - Eierklassifikation/public/Files/SC-KI_2022/train'
    TEST_DATA_PATH = '04 - Eierklassifikation/public/Files/SC-KI_2022/test'
    return (
        DataLoader,
        TEST_DATA_PATH,
        TRAIN_DATA_PATH,
        nn,
        plt,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def _(DataLoader, TEST_DATA_PATH, TRAIN_DATA_PATH, torchvision, transforms):
    IMG_SIZE = 64
    transforms_1 = transforms.Compose([transforms.Resize([IMG_SIZE, IMG_SIZE]), transforms.ToTensor(), transforms.Grayscale()])
    train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transforms_1)
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transforms_1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return test_loader, train_dataset, train_loader


@app.cell
def _(plt, train_dataset):
    # Mit dieser Methode kannst du dir ein Bild anzeigen lassen.
    def bild_anzeigen(i):
        plt.imshow( train_dataset[i][0].permute(1, 2, 0), cmap="gray" )
        print(f"Dieses Bild wird der Klasse {train_dataset[i][1]} zugeordnet")

    # Finde durch Probieren ein Ei von einer Blaumeise, einer Ente und einem Greifvogel.
    bild_anzeigen(0)
    bild_anzeigen(1)
    return


@app.cell
def _(nn, torch):
    class CNN(nn.Module):

        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, (15, 15))
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(64, 16, (4, 4))
            self.bn2 = nn.BatchNorm2d(16)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 11 * 11, 512)
            self.bn3 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 180)
            self.bn4 = nn.BatchNorm1d(180)
            self.fc3 = nn.Linear(180, 3)
            self.relu = torch.nn.ReLU()
            self.softmax = torch.nn.Softmax()

        def forward(self, _x):
            _x = self.pool1(self.relu(self.bn1(self.conv1(_x))))
            _x = self.pool2(self.relu(self.bn2(self.conv2(_x))))
            _x = _x.view(-1, 16 * 11 * 11)
            _x = self.relu(self.bn3(self.fc1(_x)))
            _x = self.relu(self.bn4(self.fc2(_x)))
            _x = self.fc3(_x)
            _x = self.softmax(_x)
            return _x
    return (CNN,)


@app.cell
def _(torch):
    def test_model(model, data):
        total = 0
        correct = 0
        for (_x, _y) in data:
            output = model(_x)
            output = torch.argmax(output, dim=1)
            correct = correct + sum(torch.eq(output, _y)).item()
            total = total + len(_y)
        return round(correct / total, 3)
    return (test_model,)


@app.cell
def _(CNN, torch):
    # Hier erzeugen wir ein Objekt des CNNs.
    cnn = CNN()

    # Füge hier eine sinnvolle Lernrate ein.
    LERNRATE = 10
    optimizer = torch.optim.SGD(cnn.parameters(), lr = LERNRATE )

    # Loss-Funktion
    loss_func = torch.nn.CrossEntropyLoss()
    return cnn, loss_func, optimizer


@app.cell
def _(cnn, test_loader, test_model, train_loader):
    # Teste zu Beginn mit Hilfe der Methode test_model, wie gut 
    # dein neuronales Netz die Trainigs- und Testdaten ohne Training klassifiziert. 
    # Gib das Ergebnis in Prozent aus.
    print(test_model(cnn, train_loader)*100,"%")
    print(test_model(cnn, test_loader)*100,"%")
    return


@app.cell
def _(cnn, loss_func, optimizer, train_loader):
    for epoche in range(5):
        loss_epoch = 0
        for (_x, _y) in train_loader:
            cnn.train(True)
            optimizer.zero_grad()
            output = cnn(_x)
            loss = loss_func(output, _y)
            loss.backward()
            optimizer.step()
            loss_epoch = loss_epoch + loss
        print(loss_epoch.item())
    return


@app.cell
def _(cnn, test_loader, test_model, train_loader):
    # Gib hier die Genauigkeit auf den Trainings- und Testdaten aus.
    print(test_model(cnn, train_loader)*100,"%")
    print(test_model(cnn, test_loader)*100,"%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ____

    <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>

    <i>Gib jeweils zehn Bilder der Testdaten aus, die richtig und falsch klassifiziert wurden.</i>
    """
    )
    return


@app.cell
def _(cnn, plt, test_loader, torch):
    def show(_x, _y, i):
        print('Klasse', output[i].item(), '/', _y[i].item(), '-', correct[i].item())
        plt.imshow(_x[i].permute(1, 2, 0), cmap='gray')
        plt.show()
    corr = 0
    incorr = 0
    for (_x, _y) in test_loader:
        output = cnn(_x)
        output = torch.argmax(output, dim=1)
        correct = torch.eq(output, _y)
        for i in range(0, len(correct)):
            if correct[i] and corr < 10:
                show(_x, _y, i)
                corr = corr + 1
            if corr >= 10:
                break
    for (_x, _y) in test_loader:
        output = cnn(_x)
        output = torch.argmax(output, dim=1)
        correct = torch.eq(output, _y)
        for i in range(0, len(correct)):
            if not correct[i] and incorr < 10:
                show(_x, _y, i)
                incorr = incorr + 1
            if incorr >= 10:
                break
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ____

    <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>

    <i>Erreicht dein neuronales Netz eine gute Genauigkeit? Falls ja, hast du den Code richtig ergänzt. Jetzt kannst du versuchen, dein neuronales Netz zu optimieren. Du kannst z.B. neue Schichten einfügen, die Anzahl der Inputs/Outputs ändern oder eine andere Lernrate ausprobieren. Beachte dabei auch die untere Grafik.</i>

    <figure>
      <img src="public/resources/img/overfitting.png" alt="Overfitting" style="width:50%">
    </figure>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Bilderverzeichnis

    https://tenor.com/de/view/kermit-worried-oh-no-anxious-gif-11565777

    https://i0.wp.com/developersbreach.com/wp-content/uploads/2020/08/cnn_banner.png?fit=1400%2C658&#038;ssl=1
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
