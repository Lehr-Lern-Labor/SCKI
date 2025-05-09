import marimo

__generated_with = "0.12.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Einführung""")
    return


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
