# Hinweise zu den Lösungen
## Hervorhebungen im Code
Im Code werden Lösungen immer mit

```Python
# ---------- Lösung

# -----------------
```

markiert. Teile der Aufgabenstellung, die für die Lösung entfernt werden müssen, werden mit `##` auskommentiert, sodass in der Lösung immer noch ersichtlich ist, wie die ursprüngliche Aufgabenstellung aussieht.

## Umgang mit Git
Jupyter Notebooks sind nur bedingt für die Verwaltung mit Code geeignet. Unter Umständen kann es passieren, dass die Änderungen im `student`-Branch manuell auf den `teacher`-Branch übernommen werden müssen, da der automatische Merge nicht funktioniert. Hier bietet es sich an, die Änderungen außerhalb von Git zu übernehmen und dann im Merge über

```
git checkout <File> --ours
git add .
git commit
```
die vorgeschlagenen Änderungen zu überschreiben.

# Die virtuelle Maschine
Die VM basiert auf Ubuntu 24.04 (ubuntu-24.04-desktop-amd64.iso) und arbeitet mit [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) (hier ist jupyter notebook bereits integriert).

## Initialisierung
```bash
$ sudo apt-get update
$ sudo apt install curl
$ cd /tmp/
$ curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
$ bash Anaconda3-2024.06-1-Linux-x86_64.sh
# > yes > yes
$ source ~/.bashrc
```

## Repo & Skripte einrichten
```bash
$ cd ~
$ sudo apt install git
```

`~/LLL-utils/scki.sh`
```bash (scki.sh)
#!/bin/bash
cd ~/Schreibtisch
if [ ! -d "SCKI" ]; then
	git clone -b current --single-branch https://github.com/Lehr-Lern-Labor/SCKI.git SCKI
fi
cd SCKI
git pull
source /home/lll/anaconda3/bin/activate
jupyter notebook
echo "Press any key to exit..."
read -s -n 1
```

```bash
# Alle branches anzeigen
$ git remote set-branches origin "*"
```

`~/Schreibtich/SCKI.desktop`
```bash
[Desktop Entry]
Type = Application
Name = SCKI
Exec = /home/lll/LLL-utils/scki.sh
Icon = /home/lll/LLL-utils/favicon.png
Terminal = true
```

## Pakete installieren
```
$ pip install torch
$ pip install torchvision
$ pip install gym
$ pip install genetics
$ pip install tensorboardX
$ pip install pygame
```

```
# Delete problematic files:
$ rm ~/anaconda3/lib/libstdc++.so
$ rm ~/anaconda3/lib/libstdc++.so.6
$ rm ~/anaconda3/lib/libstdc++.so.6.0.29
```

## Delete all output from terminal
```
$ cat /dev/null > ~/.bash_history && history -c && exit
```

## Shrink VM ([Anleitung](https://gist.github.com/kuznero/576e848c39080745ac1915c6b3e4820b))
```bash
# On Guest
$ sudo apt install pv
$ sudo dd if=/dev/zero | pv | sudo dd of=/bigemptyfile bs=4096k
$ sudo rm -rf /bigemptyfile

# Shutdown Guest

# On Host
# Windows: VBoxManage.exe modifyhd c:\path\to\thedisk.vdi --compact
$ "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" modifyhd "C:\Users\<user>\VirtualBox VMs\LLL\LLL.vdi" --compact

# Linux: vboxmanage modifyhd /path/to/thedisk.vdi --compact
# Mac: VBoxManage modifyhd /path/to/thedisk.vdi --compact

# Appliance exportieren
```
