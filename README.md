# Learning #
<pre>
some simple scripts to help me to understand algorithms...
einige einfache Skripte um Algorithmen zu verstehen...
</pre>
## Installation und Benutzung mit venv (Linux/Mac) ##
<pre>
python3 -m venv path/to/venv
source path/to/venv/bin/activate
python3 -m pip install networkx>3 matplotlib>3 numpy>2 pandas>2 openpyxl>3 graphviz
sudo apt install graphviz
python3 ID3.py
</pre>
## Installation und Benutzung mit venv (Windows) ##
<pre>
python3 -m venv path\to\venv
.\path\to\venv\Scripts\activate
python3 -m pip install networkx>3 matplotlib>3 numpy>2 pandas>2 openpyxl>3 graphviz
python3 ID3.py
</pre>
## Installation und Benutzung ohne venv ##
<pre>
python3 -m pip install networkx>3 matplotlib>3 numpy>2 pandas>2 openpyxl>3 graphviz
python3 ID3.py
</pre>
### ID3.py ###
<pre>
Takes a CSV or XLSX as source, shows how to calculate the
ID3 tree and print it out as PDF.
Akzeptiert als Quelle eine CSV oder Excel-Datei und zeigt
den Ablauf der Berechnung eines ID3-Baums für diese Daten
Erstellt ein PDF mit dem berechneten Baum.
</pre>
### ID3_nxtree.py ###
<pre>
Variante (für Windows) mit matplotlib und networkx,
falls graphviz nicht verwendbar ist...
</pre>
### NaiveBayes.py ###
<pre>
Takes a CSV or XLSX as source, shows how to calculate the
Likelihoods with Naive Bayes 
Akzeptiert als Quelle eine CSV oder Excel-Datei und zeigt
den Ablauf der Berechnung der Naive Bayes Wahrscheinlichkeit
</pre>
