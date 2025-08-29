# Learn Datamining

Einige einfache Skripte, um Datamining Algorithmen besser zu verstehen.  
Hier auch als Flask-App zum Testen verfügbar (ID3, NaivBayes, kNN):  
👉 [https://dm.dhn.network/](https://dm.dhn.network/)

---

## 🔧 Installation und Benutzung

### Mit `venv` unter Linux/Mac

```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
python3 -m pip install networkx matplotlib numpy pandas openpyxl graphviz
sudo apt install graphviz
python3 ID3.py
```

### Mit `venv` unter Windows

```cmd
python3 -m venv path\to\venv
.\path\to\venv\Scripts\activate
python3 -m pip install networkx matplotlib numpy pandas openpyxl graphviz
python3 ID3.py
```

---

## 🧠 Skripte

### `ID3.py`

- Zeigt Schritt für Schriit den Ablauf der Berechnung eines **ID3-Entscheidungsbaums**.
- Akzeptiert eine CSV- oder Excel-Datei als Datenquelle.
- Erstellt eine PDF-Datei mit dem berechneten Baum.
- Nutzt Graphviz zur Visualisierung.

### `ID3_nxtree.py`

- Alternative Version für **Windows**, wenn Graphviz nicht verfügbar ist.
- Visualisierung mit **matplotlib** und **networkx** statt Graphviz.

### `NaiveBayes.py`

- Zeigt die Berechnung der **Naive-Bayes-Wahrscheinlichkeiten** Schritt für Schritt.
- Akzeptiert eine CSV- oder Excel-Datei als Datenquelle.

---

## 📦 Graphviz unter Windows

Um Graphviz unter Windows nutzen zu können:

1. Download: [https://graphviz.org/download](https://graphviz.org/download)
2. Installationspfad (z. B. `C:\Program Files\Graphviz\bin`) zur Systemumgebungsvariable **`PATH`** hinzufügen

---
