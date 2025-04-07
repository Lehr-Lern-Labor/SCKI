import marimo

__generated_with = "0.12.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Einführung""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Der Umgang mit Jupyter Notebooks""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Code ausführen""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In diesem Notebook kannst du Python-Code direkt ausführen. Entsprechende Code-Blöcke sind grau hinterlegt. 

        ---
        ✎ *Führe den folgenden Block aus, indem du das Feld auswählst und entweder oben in der Leiste auf* **[▶](## "Run this cell and advance")** *klickst oder* **[⇧](## "Shift") + [⏎](## "Enter")** *drückst.*  
        🗬 *Mit* **[Strg](## "Strg") + [⏎](## "Enter")** *kannst du einen Block ausführen ohne zum nächsten Block weiterzuspringen.*
        """
    )
    return


@app.cell
def _():
    # Das ist ein ausführbares Feld.
    # Zeilen, die mit dem Symbol # beginnen, sind Kommentare und werden vom Compiler ignoriert. 

    print("Herzlich Willkommen im Lehr-Lern-Labor Informatik am KIT!")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Den Python-Code könntest du einfach in ein Python-File kopieren und von dort ausführen. Jupyter Notebooks ermöglichen es jedoch, einzelne Code-Blöcke getrennt von anderen auszuführen. Das hat den Vorteil, dass du dich auf einzelne Code-Abschnitte konzentrieren kannst, kann aber unter Umständen auch dazu führen, dass sich dein Code unerwartet verhält, weil der Code nicht von oben nach unten ausgeführt wurde.

        🗬 *Die Zahlen in den eckigen Klammern links vom Block verraten dir, dass und in welcher Reihenfolge der Code ausgeführt wurde.*

        ---
        ✎ Führe die folgenden Code-Blöcke von oben nach unten aus. Welche Zahl wird am Ende ausgegeben?
        """
    )
    return


@app.cell
def _():
    a = 1
    return (a,)


@app.cell
def _(a):
    a_1 = a + 3
    return (a_1,)


@app.cell
def _(a_1):
    a_2 = a_1 * 10
    return (a_2,)


@app.cell
def _(a_2):
    a_3 = a_2 + 2
    return (a_3,)


@app.cell
def _(a_3):
    print(a_3)
    return


@app.cell
def _():
    a_4 = 1
    a_4 = a_4 + 3
    a_4 = a_4 * 10
    a_4 = a_4 + 2
    print(a_4)
    a_4 = 1
    a_4 = a_4 * 10
    a_4 = a_4 * 10
    a_4 = a_4 + 3
    a_4 = a_4 + 3
    a_4 = a_4 + 3
    a_4 = a_4 + 2
    print(a_4)
    return (a_4,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ✎ *Nutze die obigen Code-Blöcke ohne Änderung, um `111` auszugeben.*
        <details>
            <summary>Brauchst du Hilfe?</summary>
            Du kannst die Blöcke mehrfach und in beliebiger Reihenfolge ausführen.
        </details>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Kernel neu starten""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Wenn sich dein Programm unerwartet verhält, kannst du den *Kernel* neu starten. Dadurch wird der aktuelle Zustand des bereits ausgeführten Codes verworfen und du kannst bei der Ausfürhung von Code wieder bei null beginnen.

        Um den Kernel neu zu starten hast du verschiedene Möglichkeiten:
        * **Klicke im Menü auf [Kernel > Restart Kernel and Run up to Selected Cell](##).**  
          Dadurch wird der Kernel neu gestartet und der gesamte Code bis zum ausgewählten Block wieder ausgeführt.
        * **Klicke im Menü auf [Kernel > Restart Kernel and Clear Output of All Cells](##).**  
          Dadurch wird der Kernel neu gestartet und die bisherigen Ausgaben werden gelöscht. Den Code musst du dann selbst wieder ausführen.
        * **Klicke oben in der Leiste auf [⟳](## "Restart the kernel").**  
          Dadurch wird der Kernel neu gestartet, die bisherigen Ausgaben bleiben aber erhalten. Den Code musst du dann selbst wieder ausführen.
        * **Klicke oben in der Leiste auf [▶▶](## "Restart the kernel and run all cells").**  
          Dadurch wird der Kernel neu gestartet und der gesamte Code von oben nach unten ausgeführt - auch unterhalb des aktuellen Blocks.

        Welche Variante du verwendest, bleibt dir überlassen und hängt unter Umständen von der jeweiligen Situation ab. Wichtig ist, dass du weißt, wie man den Kernel neu startet und was dadurch passiert.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Speichern""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Um deine Änderungen nicht zu verlieren, solltest du das Notebook regelmäßig speichern. Dabei wird nicht nur dein Code gespeichert sondern  auch die zugehörigen Ausgaben. Der Zustand des Kernels wird allerdings nicht gespeichert, sodass du den Code das nächste Mal am besten wieder von ganz oben aus ausführst. 

        Das Notebook kannst du genauso wie in anderen Editoren speichern, z.B. durch Klicken auf **[🖫](## "Save and create checkpoint")**, durch Drücken von **[Strg + S](##)** oder im Menü über **[File > Save Notebook](##)**.

        ---
        ✎ *Speichere das Notebook.*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Navigation & Übersichtlichkeit""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Inhaltsverzeichnis
        Im Menü kannst du über **[View > Table of Contents](##)** (oder alternativ mit **[Strg + ⇧ + K](## "Strg + Shift + K")**) das Inhaltsverzeichnis anzeigen lassen. Darüber kannst du schneller im gesamten Dokument navigieren und hast einen Überblick über die gesamte Struktur des Notebooks.

        🗬 *Wenn du im Inhaltsverzeichnis links oben auf **[≡](## "Show heading number in the document")** klickst, kannst du dir die Überschriften nummeriert anzeigen lassen.*

        #### Zeilennummern
        Im Menü kannst du über **[View > Show Line Numbers](##)** Zeilennummern anzeigen lassen. Das hilft dabei, über spezielle Stellen im Code zu sprechen.

        #### Inhalte einklappen
        Wenn du mit der Maus über eine Überschrift fährst, erscheint links ein kleiner Pfeil. Damit kannst du ganze Abschnitte einklappen, wenn du sie gerade nicht benötigst und dann auch wieder ausklappen. Beispielweise könntest du das gesamte Kapitel über den Umgang mit Jupyter Notebook einklappen, um das Notebook kürzer zu machen und dich besser auf die folgenden Aufgaben konzentrieren zu können - aber natürlich nur, wenn dir jetzt klar ist, wie du mit einem Jupyter Notebook umgehst.

        🗬 *Übrigens werden eingeklappte Inhalte auch im Inhaltsverzeichnis entsprechend angezeigt. Du kannst daher auch über das Inhaltsverzeichnis Inhalte ein- und auch wieder ausklappen.*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Warum Python?""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Python ist eine Programmiersprache, die in den letzten Jahren immer beliebter wurde. Python lässt sich aufgrund der einfachen Syntax schnell erlernen und bietet insbesondere zu aktuellen Themen der künstlichen Intelligenz wie beispielsweise *tiefen neuronalen Netzen* oder *Sprachverarbeitung* (Natural Language Processing) eine breite Auswahl an Bibliotheken (d.h. Code den man für seine eigenen Projekte benutzen kann ohne alles selbst schreiben zu müssen). Laut PYPL-Index ("PopularitY of Programming Language Index") ist Python momentan sogar die beliebteste Programmiersprache weltweit. 

        <img src="public/resources/img/statista_beliebteste_programmiersprachen.png" alt="Liste der beliebtesten Programmiersprachen" style="width:50%">
        <!-- Quelle: https://de.statista.com/statistik/daten/studie/678732/umfrage/beliebteste-programmiersprachen-weltweit-laut-pypl-index/ -->

        Auch für unsere Anwendungsfälle eignet sich Python, weshalb du mit den Grundlagen der Sprache vertraut sein solltest.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Syntax""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Einrückungen""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Python arbeitet mit Einrückungen, um die Struktur des Programms sichtbar zu machen. Aus anderen Programmiersprachen wie beispielsweise *C#* oder *Java* kennst du vielleicht die geschweiften Klammern, die dort stattdessen verwendet werden. Das bedeutet aber auch, dass du deinen Code nicht beliebig einrücken darfst. Achte also im Folgenden besonders darauf, welcher Code eingerückt werden muss.

        ✎ *Führe den folgenden Code aus und behebe anschließend den Fehler.*
        """
    )
    return


@app.cell
def _():
    # Hier wurde richtig eingerückt.
    good = "Hallo!"
    print(good)

    # Diese Einrückung ist falsch
    bad = "Hi!"
    ##    print(bad)
    # ---------- Lösung
    print(bad)
    # -----------------
    return bad, good


@app.cell
def _(mo):
    mo.md(r"""### Variablen & Datentypen""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In Python ist es denkbar einfach, eine Variable zu definieren:
        ```python
        name = wert
        ```

        Der `name` einer Variablen kann frei gewählt werden, sollte aber aussagekräftig sein - das verbessert die Lesbarkeit des Codes.

        Und auch der `wert` einer Variable kann frei gewählt werden. Im Gegensatz zu anderen Programmiersprachen musst du den Datentyp einer Variablen (z.B. Zahl, Wahrheitswert oder Zeichenkette) nicht im Vorhinein festlegen. Stattdessen wird während der Ausführung des Programms entschieden, welchen Datentyp die Variable haben sollte.

        ```python
        zahl        = 42
        dezimalzahl = 4.2
        wahr        = True
        falsch      = False
        string      = "Eine Zeichenkette in Anführungszeichen"
        ```

        ---
        ✎ *Ergänze die fehlenden Variablen, sodass die Ausgabe fehlerfrei ist.*
        """
    )
    return


@app.cell
def _():
    steckbrief = "Steckbrief:"

    # Füge hier deinen Code ein.
    # ---------- Lösung
    name = "Lehr-Lern-Labor"
    maennlich = False
    alter = 5
    groesse = 42
    # -----------------

    print(steckbrief, "\n", 
          "Name:", name, "\n", 
          "männlich:", maennlich, "\n", 
          "Alter:", alter, "\n", 
          "Größe (in m):", groesse)
    return alter, groesse, maennlich, name, steckbrief


@app.cell
def _(mo):
    mo.md(r"""# Erfahrung sammeln""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Variablen""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Du kannst Variablen deklarieren, ihnen Werte zuweisen, sie miteinander kombinieren und sie ausgeben lassen. Dabei helfen dir die folgenden Beispiele.

        ### Zahlen
        ```Python
        x = 42                      # deklarieren und initialisieren (d.h. den ersten Wert zuweisen)
        x = 2                       # Wert zuweisen

        # Rechnen
        a = x + 1                   # Addition
        b = x - 2                   # Subtraktion
        c = x * 3                   # Multiplikation
        d = x / 4                   # Division
        e = x ** 5                  # Potenz
        f = x // 6                  # Ganzzahlige Division
        g = x % 7                   # Modulo (Rest der ganzzahligen Division)

        # Kurzschreibweisen
        x += 2                      # x = x + 2
        x -= 2                      # x = x - 2
        x *= 2                      # x = x * 2
        x /= 2                      # x = x / 2
        x **= 2                     # x = x ** 2

        # Zahlen ausgeben
        print(x)
        ```
        ### Zeichenketten
        ```Python
        hello = "Hallo"            # Zeichenketten, also Strings, deklarieren
        world = 'Welt'             # "" und '' können beide verwendet werden

        greeting = hello + world   # Konkatenation (d.h. Zeichenketten aneinanderhängen)
        length = len(greeting)     # Länge einer Zeichenkette
        a = greeting[0]            # Erstes Zeichen (die Nummerierung beginnt bei 0)
        b = greeting[-1]           # Letztes Zeichen (negative Nummern zählen von hinten)
        c = greeting[3:6]          # Teil der Zeichenkette (einschließlich Zeichen Nummer 3, ausschließlich Zeichen Nummer 6)
        d = greeting[:5]           # d = greeting[0:5] (Erster Teil der Zeichenkette, die ersten 5 Zeichen)
        e = greeting[5:]           # e = greeting[5:-1] (Letzter Teil der Zeichenkette, ab Zeichen Nummer 5)

        # Zeichenketten ausgeben
        print(greeting)
        print(hello, world)
        print(hello, "an die gesamte", world, "!")

        # 🗬 Zeichenketten besser formatieren (Weitere Infos unter https://docs.python.org/3/tutorial/inputoutput.html)
        print(f"{hello} an die gesamte {world}!")
        ```

        ### Wahrheitswerte
        ```Python
        wahr = True
        falsch = False

        # Boolsche Operatoren
        a = not wahr               # Verneinung
        b = wahr and falsch        # Und
        c = wahr or falsch         # Oder

        # Vergleichen
        d = (wahr == falsch)       # Gleichheit
        e = (wahr != falsch)       # Ungleichheit
        f = (1 < 2)                # Kleiner
        g = (1 <= 2)               # Kleiner oder gleich
        h = (1 > 2)                # Größer
        i = (1 >= 2)               # Größer oder gleich
        j = (0 <= x < 42)          # Zahlenbereich
        ```

        ---
        ✎ *Schau dir diese Beispiele an und stelle sicher, dass du weißt, was der Code bewirkt. Kopiere bei Bedarf Code in das untenstehende Feld und führe ihn aus.*
        """
    )
    return


@app.cell
def _():
    print('------------------ Zahlen')
    _x = 42
    _x = 2
    a_5 = _x + 1
    b = _x - 2
    c = _x * 3
    d = _x / 4
    e = _x ** 5
    f = _x // 6
    g = _x % 7
    _x = _x + 2
    _x = _x - 2
    _x = _x * 2
    _x = _x / 2
    _x = _x ** 2
    print(_x)
    print(a_5, b, c, d, e, f, g)
    print('------------------ Zeichenketten')
    hello = 'Hallo'
    world = 'Welt'
    greeting = hello + world
    a_5 = greeting[0]
    b = greeting[-1]
    c = greeting[3:6]
    d = greeting[:5]
    e = greeting[5:]
    print(greeting)
    print(hello, world)
    print(hello, 'an die gesamte', world, '!')
    print(f'{hello} an die gesamte {world}!')
    print(a_5, b, c, d, e)
    print('------------------ Wahrheitswerte')
    wahr = True
    falsch = False
    a_5 = not wahr
    b = wahr and falsch
    c = wahr or falsch
    d = wahr == falsch
    e = wahr != falsch
    f = 1 < 2
    g = 1 <= 2
    h = 1 > 2
    _i = 1 >= 2
    j = 0 <= _x < 42
    print(a_5, b, c, d, e, f, g, h, _i, j)
    return a_5, b, c, d, e, f, falsch, g, greeting, h, hello, j, wahr, world


@app.cell
def _(mo):
    mo.md(r"""✎ *Ergänze den folgenden Code. Die Funktion `pruefe_eingabe()` verrät dir, ob du alles richtig gemacht hast.*""")
    return


@app.cell
def _():
    from public.resources.code.help_functions import pruefe_eingabe
    _x = 9
    _x = _x + 6
    _x = _x * 3
    _x = _x ** 4
    _x = _x / 91125
    _x = _x - 3
    pruefe_eingabe(_x)
    return (pruefe_eingabe,)


@app.cell
def _(mo):
    mo.md(r"""✎ *Ergänze den folgenden Code. Überprüfe dein Ergebnis selbst.*""")
    return


@app.cell
def _():
    text = "Labor"
    string = "Lehr"
    zeichenkette = "Lern"

    # Konkateniere die drei Zeichenketten in der richtigen Reihenfolge und speichere das Ergebnis in der Variable LLL
    # ---------- Lösung
    LLL = string + zeichenkette + text
    # -----------------

    # Gib das erste Zeichen von LLL aus, das Zeichen Nummer 4 und das Zeichen Nummer 8.
    # ---------- Lösung
    print(LLL[0],LLL[4],LLL[8])
    # -----------------

    # Gib nun die drei Bestandteile des Worts aus.
    # Verwende dafür aber nicht die zu Beginn definierten Variablen, sondern nur die Variable LLL.
    # ---------- Lösung
    print(LLL[:4], LLL[4:8], LLL[-5:])
    # -----------------

    # Verwende nun zusätzliche Zeichen, um den vollständigen Namen auszugeben.
    # ---------- Lösung
    print(f"{string}-{zeichenkette}-{text} Informatik Karlsruhe")
    print(string+"-"+zeichenkette+"-"+text, "Informatik", "Karlsruhe")
    # -----------------
    return LLL, string, text, zeichenkette


@app.cell
def _(mo):
    mo.md(r"""## Funktionen""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Um Code nicht mehrfach schreiben zu müssen, verwendet man Funktionen, die an beliebigen Stellen im Programm aufgerufen werden können.

        Funktionen besitzen in Python den folgenden Aufbau:
        ```Python
        # Funktion definieren
        def funktion(parameterA, parameterB):
            tuEtwas()
            return rückgabe

        # Funktion aufrufen
        funktion(wertA, wertB)
        ```
        Du kannst den Namen der Funktion (fast) frei wählen und musst bei der Definition einer Funktion nicht festlegen, ob ein Wert zurückgegeben wird oder nicht. Falls du keinen Wert zurückgeben möchtest, lass die `return`-Zeile einfach weg.

        ---
        ✎ *Ergänze die Funktion sinnvoll und begrüße dich selbst.*
        """
    )
    return


@app.cell
def _():
    def sageHallo(name):
        # Füge hier deinen Code ein und lösche 'pass'.
        ##pass
        # ---------- Lösung
        print("Hallo " + name)

        # alternativ: return "Hallo " + name

        # -----------------

    sageHallo("Kim")
    sageHallo("Lehr-Lern-Labor")
    return (sageHallo,)


@app.cell
def _(mo):
    mo.md(r"""## Kontrollstrukturen""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Bedingungen (if / else)
        Bedingungen besitzen in Python den folgenden Aufbau:
        ```Python
        if (bedingungA):
            tuEtwas()
        elif (bedingungB):
            tuEtwasAnderes()
        else:
            tuSonstEtwas()
        ```
        Dabei brauchst du weder den `elif`- noch den `else`-Block. Und auch auf die Klammern um die Bedingung kannst du verzichten.

        ---
        ✎ *Realisiere folgenden Sachverhalt:*

        Wenn der eingeloggte Benutzer Jana ist, dann soll sie zunächst begrüßt werden. Wenn sie das Passwort richtig eingeben hat, soll ihr aktueller Punktestand von 2241 Punkten ausgegeben werden. Wenn sie das Passwort falsch eingegeben hat, soll eine entsprechende Rückmeldung ausgegeben werden. Wenn der eingeloggte Benutzer Ramon ist, dann ist der Ablauf analog zum oberen Fall, wobei sein aktueller Punktestand 2110 beträgt. Wenn der eingeloggte Benutzer weder Jana noch Ramon ist, dann soll ausgegeben werden, dass der Nutzername nicht bekannt ist.
        """
    )
    return


@app.cell
def _():
    def getScore(name, isPasswordCorrect):
        # Füge hier deinen Code ein und lösche 'pass'.
        ##pass
        # ---------- Lösung
        if name == "Jana":
            print("Hallo", name)
            if isPasswordCorrect:
                print("Score:", 2241)
            else:
                print("Das Passwort stimmt leider nicht.")
        elif name == "Ramon":
            print("Hallo", name)
            if isPasswordCorrect:
                print("Score:", 2110)
            else:
                print("Das Passwort stimmt leider nicht.")
        else:
            print("Der Nutzername",name, "ist nicht bekannt.")
        # -----------------

    getScore("Jana", True)
    getScore("Ramon", False)
    getScore("Kim", True)
    return (getScore,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Schleifen

        #### while-Schleife
        Der Code in der while-Schleife wird solange ausgeführt, bis die Bedingung der while-Schleife nicht mehr erfüllt ist. Wenn die Bedingung nie auf *falsch* gesetzt wird, dann wird die while-Schleife unendlich oft ausgeführt und dein Programm wird nie regulär beendet. 

        While-Schleifen besitzen in Python den folgenden Aufbau:

        ```Python
        while bedingung:
            tuEtwas()
        ```

        ---
        ✎ *Berechne mithilfe einer while-Schleife die Summe der Zahlen von 1 bis 10.*
        """
    )
    return


@app.cell
def _():
    sum = 0
    _i = 0
    while _i <= 10:
        sum = sum + _i
        _i = _i + 1
    print(sum)
    return (sum,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### for-Schleife
        Möchtest du über einen Zahlenbereich (ganze Zahlen), eine Zeichenkette oder allgemein über ein iterierbares Objekt iterieren, so eignet sich die for-Schleife dazu.

        For-Schleifen besitzen in Python den folgenden Aufbau:
        ```Python
        for element in iterierbaresObjekt:
            tuEtwas(element)
        ```
        Häufig möchte man dabei mit einer Zählvariablen arbeiten. Dabei hilft die `range`-Funktion.
        ```Python
        for zähler in range(start, stop, schritt):
            tuEtwas(zähler)

        range(2, 42, 2)     # Zähle von 2 bis ausschließlich 42 in Zweier-Schritten
        range(1, 30)        # Zähle von 1 bis ausschließlich 30 (in Einser-Schritten)
        range(5)            # Zähle von 0 bis ausschließlich 5 (also 5 Zahlen) 
        ```

        ---
        ✎ *Zähle mit Hilfe einer for-Schleife die Vorkommen des Bustaben `a` im vorgegebenen Satz. Anschließend kannst du mit `satz.count('a')` überprüfen, ob du richtig liegst.*
        """
    )
    return


@app.cell
def _():
    satz = 'Max wachst Wachsmasken. Was macht Max? Wachsmasken wachst Max.'
    anzahl = 0
    for _buchstabe in satz:
        if _buchstabe == 'a':
            anzahl = anzahl + 1
    print(anzahl)
    print(anzahl == satz.count('a'))
    return anzahl, satz


@app.cell
def _(mo):
    mo.md(r"""## Daten""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Tupel

        In Tupeln kannst du mehrere Informationen speichern. Einmal definiert, können die Werte in einem Tupel nicht mehr verändert werden.

        ```Python
        # Tupel definieren
        tupel = (wertA, wertB, wertC)

        # Tupel auslesen
        tupel[0]     # erstes Element
        tupel[-1]    # letztes Element

        len(tupel)   # Länge des Tupel
        ```

        ---
        ✎ *Gib für jede Person den Namen und den Beruf aus.*
        """
    )
    return


@app.cell
def _():
    sheldon = ('Sheldon', 'Cooper', '26. Februar 1980', 'Theoretischer Teilchenphysiker')
    leonard = ('Leonard', 'Hofstadter', '17. Mai 1980', 'Experimentalphysiker')
    amy = ('Amy', 'Farrah Fowler', '17. Dezember', 'Neurobiologin')

    # Füge hier deinen Code ein.
    # ---------- Lösung
    for person in [sheldon, leonard, amy]:
        print(person[0] + " " + person[1] + " ist " + person[3] + ".")
    # -----------------
    return amy, leonard, person, sheldon


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Listen

        Du kannst mit Hilfe von Listen mehrere Elemente speichern, so wie du es oben bei Tupeln gesehen hast. Allerdings kannst du die Elemente von Listen verändern, löschen und neue Elemente hinzufügen.

        ```Python
        # Liste definieren und erweitern
        list = [wertA, wertB, wertC]     # Liste definieren
        otherList = [wert1, wert2, wer3]

        list.append(wertD)               # Element hinten hinzufügen
        list = list + wertE              # Element zu Liste hinzufügen

        newList = list + otherList       # Zwei Listen aneinanderhängen
        zip(list, otherList)             # Hängt Paare aus Elementen aus beiden Listen aneinander: [[wertA,wert1], ...]

        # Liste auslesen
        list[0]                          # erstes Element
        list[-1]                         # letztes Element

        len(list)                        # Länge der Liste

        erstesElement = list.pop()       # Erstes Element entfernen und zurückgeben
        zweitesElement = list.pop(1)     # Zweites Element entfernen und zurückgeben

        ```
        Bei Listen wird in der Regel nur die *Referenz* kopiert, d.h. wenn man in der "Kopie" etwas verändert, wird auch das Original verändert. Um das zu verhindern, muss man die Liste gezielt kopieren.
        ```Python
        list[0] = neuerWert              # Element verändern
        newList = list                   # Referenz kopieren (newList[0] = neuerWert ändert auch list[0])
        newList = list[:]                # Liste kopieren
        ```

        ---
        ✎ *Ergänze die Liste der Fibonacci-Zahlen mit den ersten 15 Fibonacci-Zahlen. Überprüfe dein Ergebnis mit der Funktion `pruefe_fibonacci()`*

        Eine Fibonacci-Zahl ist jeweils die Summe ihrer beiden Vorgänger, wobei die erste Zahl 0 und die zweite Zahl 1 ist. In der Liste `fibonacci` sind die ersten beiden Zahlen bereits abgespeichert.
        """
    )
    return


@app.cell
def _():
    from public.resources.code.help_functions import pruefe_fibonacci

    fibonacci = [0, 1]

    # Füge hier deinen Code ein.
    # ---------- Lösung
    while len(fibonacci) < 15:
        fibonacci.append(fibonacci[-2] + fibonacci[-1])
    # -----------------

    pruefe_fibonacci(fibonacci)
    return fibonacci, pruefe_fibonacci


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Dictionaries
        Abgesehen von Listen, Strings und Tupeln gibt es in Python noch einen weiteren wichtigen sequentiellen Datentyp: *Dictonaries* (in anderen Programmierspachen auch als *Maps* oder *Hashs* bezeichnet). Sie ermöglichen den Zugriff auf Werte, die sogenannten *values*, über den Namen, den sogenannten *key*.

        ```Python
        # Dictionary definieren und erweitern
        dictionary = {keyA : valueA, keyB : valueB, keyC : valueC}  # Dictionary definieren
        dictionary[keyD] = valueD                                   # Neues Wertepaar hinzufügen bzw. Eintrag überschreiben
        newDictionary = dict(zip(listA, listB))                     # Neues Dictionary aus zwei Listen zusammenfügen

        # Dictionary auslesen
        dictionary[keyA]                                            # Zu keyA gehörigen Wert auslesen
        dictionary.keys()                                           # Liste aller Namen
        dictionary.values()                                         # Liste aller Werte
        ```

        ---
        ✎ *Kodiere den gegebenen Ausdruck in Morse-Code und lasse deinen Morse-Code prüfen. Dekodiere anschließend den gegebenen Morse-Code.*
        """
    )
    return


@app.cell
def _():
    from public.resources.code.help_functions import morse, pruefe_kodiertes_wort, pruefe_dekodiertes_wort
    morse = morse()
    ausdruck = 'KÜNSTLICHE INTELLIGENZ'
    codiertes_wort = ''
    for _buchstabe in ausdruck:
        codiertes_wort = codiertes_wort + (morse[_buchstabe] + ' ')
    pruefe_kodiertes_wort(codiertes_wort)
    morse_code = '-. . ..- .-. --- -. .- .-.. . ... / -. . - --..'
    _ergebnis = ''
    demorse = dict(zip(morse.values(), morse.keys()))
    for symbol in morse_code.split(' '):
        _ergebnis = _ergebnis + demorse[symbol]
    _ergebnis = ''
    for morse_buchstabe in morse_code.split(' '):
        for key in morse:
            if morse[key] == morse_buchstabe:
                _ergebnis = _ergebnis + key
    pruefe_dekodiertes_wort(_ergebnis)
    return (
        ausdruck,
        codiertes_wort,
        demorse,
        key,
        morse,
        morse_buchstabe,
        morse_code,
        pruefe_dekodiertes_wort,
        pruefe_kodiertes_wort,
        symbol,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Klassen
        Objekte einer Klasse können nicht nur Daten, sogenannte *Attribute* enthalten sondern auch Funktionalitäten bereitstellen.

        ```Python
        # Klasse definieren
        class Klasse:
            # Konstruktor, wird beim Erzeugen einer Instanz aufgerufen
            def __init__(self, parameterA, parameterB):
                self.attributA = parameterA
                self.attributB = parameterB

            # Funktion, beachte die Verwendung von 'self' als ersten Parameter
            def funktion(self, parameterA):
                tuEtwas()

            def getAttributA(self):
                return self.attributA

            def setAttributB(self, neuerWert):
                self.attributB = neuerWert

        # Klasse verwenden
        instanz = Klasse(wertA, wertB)   # Klasse instanziieren, d.h. ein Objekt der Klasse erzeugen
        instanz.funktion(argument)       # Funktion aufrufen
        instanz.setAttributB(neuerWert)
        ```

        ---
        ✎ *Erstelle eine Klasse, die einen Schokoladenautomaten simuliert. Der Automat soll die Attribute `an` und `konto` haben. Folgende Methoden sollen in der Klasse implementiert werden: `anschalten()`, `ausschalten()`, `muenze_einwerfen()` und `schoki_ausgeben()`. Der Automat soll genau dann eine Schokolade ausgeben und den Kontostand um zwei Münzen verkleinern, wenn er eingeschaltet ist, mehr als zwei Münzen auf dem Kontostand sind und die Methode `schoki_ausgeben()` aufgerufen wird.*
        """
    )
    return


@app.cell
def _():
    class Schokomat:

        def __init__(self):
            self.an = False
            self.konto = 0

        def anschalten(self):
            print('> Anschalten')
            self.an = True

        def ausschalten(self):
            print('> Ausschalten')
            self.an = False

        def muenze_einwerfen(self):
            print('> Münze einwerfen')
            if self.an:
                self.konto = self.konto + 1
                print('Neues Guthaben:', self.konto)
            else:
                print('...')

        def schoki_ausgeben(self):
            print('> Schoki ausgeben')
            if self.an and self.konto >= 2:
                self.konto = self.konto - 2
                print(' __________________________ \n' + '|                \\|__|__|__|\n' + '|-----------------\\__|__|__|\n' + '|          SCHOKI  \\_|__|__|\n' + '|-------------------\\|__|__|\n' + '|____________________\\__|__|\n')
            elif self.an and self.konto < 2:
                print('Bitte erhöhen Sie Ihr Guthaben.')
            else:
                print('...')
    automat = Schokomat()
    automat.schoki_ausgeben()
    automat.muenze_einwerfen()
    automat.anschalten()
    automat.schoki_ausgeben()
    automat.muenze_einwerfen()
    automat.schoki_ausgeben()
    automat.muenze_einwerfen()
    automat.schoki_ausgeben()
    automat.ausschalten()
    return Schokomat, automat


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Import

        Möchtest du auf Code anderer Dateien oder auf deine lokalen Bibliotheken zugreifen, musst du sie vorher in dein Programm importieren. Danach kannst du auf die Funktionen des importierten Moduls zugreifen.
        """
    )
    return


@app.cell
def _():
    # Importe schreibst du üblicherweise am Anfang deines Programms auf.
    import time
    print("Heute:", time.asctime())

    # Durch den Schlüsselbegriff 'as' kannst du dem  
    # Import einen eigenen Namen geben.
    import os.path as p
    import os

    # Gibt den aktuellen Pfad aus.
    p.abspath(os.getcwd())

    # Wenn du nur etwas Bestimmtes aus einer Datei 
    # importieren möchtest, kannst du das wie folgt tun:
    from datetime import date

    # Gibt den Wochentag (als Integer) eines vorgegebenen Datums zurück.
    date(2021, 11, 2).weekday()

    # Du kannst auch deine eigenen Dateien in dein Programm importieren.
    # Wenn du eine Datei von einem Ordner in deinem aktuellen Verzeichnis
    # importieren möchtest, gibst du den relativen Pfad durch Punkte getrennt an:
    from public.resources.code.help_functions import sagHallo

    sagHallo()
    return date, os, p, sagHallo, time


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Fehlerbehandlung

        Um zu verhindern, dass dein Programm bei einem Fehler während der Ausführung abstürzt, muss du festlegen, was im Fehlerfall passieren soll.
        """
    )
    return


@app.cell
def _():
    _x = 1
    y = 0
    _ergebnis = 42
    try:
        print('Beginn der Ausführung des try-Blocks...')
        _ergebnis = _x / y
    except:
        print('Ups! Hier ist etwas schief gelaufen.')
    finally:
        print('Der finally-Block wird immer ausgeführt.')
    print(f'ergebnis = {_ergebnis}')
    return (y,)


@app.cell
def _(mo):
    mo.md(r"""# Weitere Aufgaben""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Zahlen sortieren

        ✎ *Schreibe eine Funktion, die eine beliebige Liste von Zahlen aufsteigend sortiert und die sortierte Liste ausgibt (z.B. `[3, -5, 1]` → `[-5, 1, 3]`).*
        """
    )
    return


@app.cell
def _():
    unsortierte_liste = [-2, 45, -102, 231, 89, -73, 42, 0, 99, 123, -11]

    def sort(list):
        changes = True
        while changes:
            changes = False
            for _i in range(1, len(list)):
                if list[_i - 1] > list[_i]:
                    help = list[_i]
                    list[_i] = list[_i - 1]
                    list[_i - 1] = help
                    changes = True
        print(list)
    sort([3, -5, 1])
    sort(unsortierte_liste)
    return sort, unsortierte_liste


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Primzahlen

        ✎ *Schreibe eine Funktion, die als Parameter eine natürliche Zahl `n` übergeben bekommt und alle Primzahlen bis `n` als Liste zurückgibt (z.B. `10` → `[2, 3, 5, 7]`).*
        """
    )
    return


@app.cell
def _():
    def prime(n):
        primes = []
        for _i in range(2, n):
            isPrime = True
            for j in range(2, _i - 1):
                if _i % j == 0:
                    isPrime = False
            if isPrime:
                primes.append(_i)
        return primes
    prime(10)
    return (prime,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Größter gemeinsamer Teiler

        ✎ *Schreibe eine Funktion, die den größten gemeinsamen Teiler zweier natürlicher Zahlen bestimmt (z.B. `30, 20` → `10`).*
        """
    )
    return


@app.cell
def _():
    def ggT(numA, numB):
        ggT = 1
        for _i in range(1, numA + 1):
            if numA % _i == 0 and numB % _i == 0:
                ggT = _i
        return ggT
    ggT(30, 20)
    return (ggT,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Primfaktorzerlegung

        ✎ *Schreibe eine Funktion, die eine natürliche Zahl in ihre Primfaktoren zerlegt (z.B. 60 = 2 $\cdot$ 2 $\cdot$ 3 $\cdot$ 5, also `60` → `2, 2, 3, 5`).*
        """
    )
    return


@app.cell
def _(prime):
    def primeFactor(n):
        primes = prime(n)
        primeFactors = []
        for p in primes:
            while n % p == 0:
                primeFactors.append(p)
                n = n / p
        print(primeFactors)
    primeFactor(60)
    return (primeFactor,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Permutationen einer Zahlenliste

        ✎ *Schreibe eine Funktion, die alle Permutation einer Zahlenliste zurückgibt (z.B. `[1,2,3]` → `[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]`).*
        """
    )
    return


@app.cell
def _():
    zahlen = [0, 1, 2, 3, 4, 5]

    # Füge deinen Code hier ein.
    # ---------- Lösung
    def permute(list):

        if len(list) == 2:
            return [[list[0],list[1]],[list[1],list[0]]]

        permutations = []

        for element in list:
            reducedList = list[:]
            reducedList.remove(element)

            for p in permute(reducedList):
                permutations.append([element] + p)

        return permutations

    permute([1,2,3])
    # -----------------
    return permute, zahlen


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Binäres Palindrom

        ✎ *Schreibe eine Funktion, die eine Zahl in ihre binäre Darstellung umwandelt und anschließend prüft, ob diese Binärzahl ein Palindrom (also vorwärts und rückwärts gelesen das gleiche) ist (z.B. $5_{10}=101_2$ → Palindrom / $6_{10}=110_2$ → kein Palindrom).*
        """
    )
    return


@app.cell
def _():
    def binary(n):
        binary = ''
        while n > 0:
            binary = str(n % 2) + binary
            n = n // 2
        return binary

    def palindrom(n):
        string = binary(n)
        isPalindrom = True
        for _i in range(0, len(string) // 2):
            if string[_i] != string[-_i - 1]:
                isPalindrom = False
                break
        if isPalindrom:
            print(n, 'ist ein Palindrom')
        else:
            print(n, 'ist kein Palindrom')
    palindrom(5)
    palindrom(6)
    return binary, palindrom


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Nim-Spiel

        ✎ *Bearbeite diese Aufgaben nur, wenn du bereits sehr fit in Python bist! Implementiere die Standardvariante des Nim-Spiels (siehe https://library.ethz.ch/standorte-und-medien/plattformen/virtuelle-ausstellungen/alles-ist-spiel/nim.html) für zwei Spieler:innen (ohne eine graphische Benutzeroberfläche). Stelle die aktuelle Anzahl der Streichhölzer mit `|` in der Ausgaben dar und frage bei jedem Zug den aktuellen Spieler, wie viele Streichhölzer er wegnehmen möchte. Verarbeite seine Ausgaben entsprechend. Recherchiere gegebenenfalls nach geeigneten Funktionen (und Bibliotheken).*
        """
    )
    return


@app.cell
def _():
    def nim(playerA='Spieler 1', playerB='Spieler 2'):
        nim = [4, 5, 6, 7]
        isPlaying = True
        firstPlayer = False
        while isPlaying:
            for _i in range(0, len(nim)):
                nPrint = '| ' * nim[_i]
                print(f'{_i + 1}  {nPrint:^15}')
            firstPlayer = not firstPlayer
            if firstPlayer:
                player = playerA
            else:
                player = playerB
            print('\n' + player)
            _x = -1
            while _x == -1:
                _x = int(input('> Reihe: '))
                if 0 < _x <= len(nim):
                    if nim[_x - 1] > 0:
                        n = -1
                        while n == -1:
                            n = int(input('> Anzahl:'))
                            if 0 < n <= nim[_x - 1]:
                                nim[_x - 1] = nim[_x - 1] - n
                            elif n > nim[_x - 1]:
                                print('Du kannst maximal', nim[_x - 1], 'Streichhölzer nehmen!')
                                n = -1
                            else:
                                print('Das geht nicht!')
                                n = -1
                    else:
                        print('Wähle eine andere Reihe!')
                        _x = -1
                else:
                    print('Diese Reihe existiert nicht!')
                    _x = -1
            isPlaying = False
            print('\n')
            for n in nim:
                if n > 0:
                    isPlaying = True
            if not isPlaying:
                print(player, 'gewinnt!')
    nim()
    return (nim,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
