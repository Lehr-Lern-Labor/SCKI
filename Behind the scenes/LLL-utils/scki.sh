#!/bin/bash
cd ~/Schreibtisch
if [ ! -d "SCKI" ]; then
	git clone -b current --single-branch https://github.com/Lehr-Lern-Labor/SCKI.git SCKI
fi
cd SCKI
if git pull; then
    source /home/lll/anaconda3/bin/activate
    jupyter notebook
    echo "Drücke eine Taste, um das Terminal zu beenden..."
    read -s -n 1
else
    echo ""
    echo "**************************************************"
    echo "  Fehler bei git pull - Bitte jemanden um Hilfe!"
    echo "**************************************************"
    echo ""
    $SHELL
fi
