# Learning #
<pre>
some simple scripts to help me to understand algorithms...
einige einfache Skripte um Algorithmen zu verstehen...
</pre>
## ID3.py ##
### Info ###
<pre>
Takes a CSV or XLSX as source, shows how to calculate the
ID3 tree and print it out as PDF.
Akzeptiert als Quelle eine CSV oder Excel-Datei und zeigt
den Ablauf der Berechnung eines ID3-Baums für diese Daten
Erstellt ein PDF mit dem berechneten Baum.
</pre>
## ID3_nxtree.py ##
### Info ###
<pre>
Variante mit matplotlib und networkx, falls graphviz
nicht verwendbar ist...
</pre>
## NaiveBayes.py ##
### Info ###
<pre>
Takes a CSV or XLSX as source, shows how to calculate the
Likelihoods with Naive Bayes 
Akzeptiert als Quelle eine CSV oder Excel-Datei und zeigt
den Ablauf der Berechnung der Naive Bayes Wahrscheinlichkeit
</pre>
### Installation ###
<pre>
python3 -m venv path/to/venv
source path/to/venv/bin/activate
pip install numpy pandas graphviz
</pre>
### Usage ###
<pre>
source path/to/venv/bin/activate
python3 ID3.py
python3 NaiveBayes.py
</pre>
