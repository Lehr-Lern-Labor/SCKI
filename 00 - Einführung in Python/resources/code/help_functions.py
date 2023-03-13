def pruefe_eingabe(x):
	if x == 42:
		print("Du hast alles richtig gemacht! Das Ergebnis ist 42")
	else:
		print("Irgendwo hat sich ein Fehler eingeschlichen. Prüfe deinen Code.")
		
def pruefe_fibonacci(liste):
	if liste == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]:
		print("Super! Das sind die ersten 15 Fibonacci-Zahlen:", liste)
	else:
		print("Irgendwo hat sich ein Fehler eingeschlichen. Probiere es noch einmal.")
		
def morse():
	morse = {
	"A" : ".-", 
	"B" : "-...", 
	"C" : "-.-.", 
	"D" : "-..", 
	"E" : ".", 
	"F" : "..-.", 
	"G" : "--.", 
	"H" : "....", 
	"I" : "..", 
	"J" : ".---", 
	"K" : "-.-", 
	"L" : ".-..", 
	"M" : "--", 
	"N" : "-.", 
	"O" : "---", 
	"P" : ".--.", 
	"Q" : "--.-", 
	"R" : ".-.", 
	"S" : "...", 
	"T" : "-", 
	"U" : "..-", 
	"V" : "...-", 
	"W" : ".--", 
	"X" : "-..-", 
	"Y" : "-.--", 
	"Z" : "--..",
	"Ä" : ".-.-",
	"Ö" : "---.",
	"Ü" : "..--",
	"0" : "-----", 
	"1" : ".----", 
	"2" : "..---", 
	"3" : "...--", 
	"4" : "....-", 
	"5" : ".....", 
	"6" : "-....", 
	"7" : "--...", 
	"8" : "---..", 
	"9" : "----.", 
	"." : ".-.-.-", 
	"," : "--..--",
	" " : "/"
	}
	return morse
    
def pruefe_kodiertes_wort(string):
    loesung = '-.- ..-- -. ... - .-.. .. -.-. .... . / .. -. - . .-.. .-.. .. --. . -. --..'
    # Entferne hintere und fordere Leerzeichen.
    string = string.strip()
    if string == loesung:
        print(f"1.Aufgabe: Richtig! Der Morsecode lautet: \n {string} \n")
    else:
        print("1.Aufgabe: Irgendwo hast du einen Fehler gemacht. Versuche es noch einmal!")
   
def pruefe_dekodiertes_wort(string):
	if string == 'NEURONALES NETZ':
		print("2.Aufgabe: Richtig! Das dekodierte Wort lautet:", string)
	else:
		print("2.Aufgabe: Irgendwo hast du einen Fehler gemacht. Versuche es noch einmal!")
		
def sagHallo():
	print("Hallo! Das ist eine importierte Funktion.")


