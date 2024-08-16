# Studienprojekt I - State Estimation Problem

Dies ist die Repository für das Studienprojekt I an der HWR. Ziel des Projekts ist die Implementierung des State Estimation Filtering Problems. Es soll sowohl parallel als auch sequentiell und in verschiedenen Programmiersprachen implementiert werden. Hierfür wurden GoLang und Python gewählt.

## Struktur und Dateien
Im Ordner "pythonImplementation" befindet sich alles was mit Python zu tun hat. Die In der "formulas.py" Date ist die eigentliche Implementierung des Filtering Problems enthalten. In der "main.py" Datei befindet sich die Matrixgeneration. Hierzu gehört sowohl die sequentielle, als auch die parallele Variante. Auch die "numbamatrixgeneration.py" enthält die Matrixgeneration, allerdings in der compilierten Python Version.

Der Ordner "golangImplementation" enthält die "main.go" Datei, in der sich die Matrizengeneration in der Sprache GoLang befindet.

## Anleitung zum Ausführen
1. Python
   - Matplotlib installieren (falls nicht schon vorhanden mit ```pip install matplotlib```)
   - Datei ausführen mit ```python <dateiname>.py```
2. Golang
   - Datei ausführen mit ```go run main.go```