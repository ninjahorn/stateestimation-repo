# Studienprojekt I und II: Vergleich einer parallelen und sequentiellen Implementierung des State Estimation Filtering Algorithmus

Dies ist die Repository für das Studienprojekt I und II an der HWR Berlin. Ziel des Projektes ist die Implementierung des State Estimation Filtering Algorithmus, sowohl parallel als auch sequentiell und in verschiedenen Programmiersprachen. Hierfür werden GoLang und Python verwendet.

## Struktur und Dateien
Im Ordner "pythonImplementation" befindet sich alles was mit Python zu tun hat. In der "filter_parallel.py" und "filter_sequential.py" Datei befindet sich die eigentliche Implementierung des Filtering Problems. In der "matrix_generation.py" Datei befindet sich die Matrixgeneration. Hierzu gehört sowohl die sequentielle, als auch die parallele Variante. Auch die "numbamatrixgeneration.py" enthält die Matrixgeneration, allerdings in der kompilierten Python Version. Für den Vergleich der beiden Filter ist die "compare_filters.py" Datei zuständig.

Der Ordner "golangImplementation" enthält die "main.go" Datei, in der sich die Matrizengeneration in der Sprache GoLang befindet.

## Anleitung zum Ausführen
1. Python
   - Matplotlib installieren (falls nicht schon vorhanden mit ```pip install matplotlib```)
   - Numpy installieren(falls nicht schon vorhanden)
   - Datei ausführen mit ```python <dateiname>.py```
2. Golang
   - Datei ausführen mit ```go run main.go```
  
## TODO:
- [ ] Strukur und Dateinamen aktualisieren
- [ ] Benötigte Installationen wie numpy oder ähnliches hinzufügen
