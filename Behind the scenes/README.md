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
