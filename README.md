# Development of design strategies for resilient technical systems using the example of a pump system 
## Master thesis of Tim Breuer

General information
* Internal number: `S417`
* Start: 14.04.2020
* Final submission deadline : 13.10.2020

## Anmerkungen für Zwischenpräsentation

* Verdana als Schriftart benutzen
* Hochaufgelöste Plots verwenden
* In Sprache nicht von "Messfehler" sondern "Messunischerheit" sprechen
* Abbildungen müssen vollständig sein, d.h. alle Parameter müssen enthalten sein
* Abbildungen / Fotos / Videos müssen eine Maßstab enthalten

## Durchführen der Monte-Carlo-Studien

* 1: Verbindung mit dem VPN herstellen
* 2: SSH-Verbindung mit dem entsprechenden Server herstellen
* 3: kopieren des Ordners ``Code`` auf den Server

in PUTTY, am besten mit dem Befehl ``screen`` einen extra Terminal erzeugen, damit die Simulation nicht abbricht, falls PUTTY die Verbindung verliert
dieser Terminal muss später detached werden: ``Ctrl``+``a`` und dann ``d``

* 4: Python-Path anpassen (muss auf den Ordner, in den der Code kopiert wurde, zeigen)
* 5: venv aktivieren, für Pakete siehe requirements.txt (vllt nicht vollständig)
* 6: Starten des Skripts der Monte-Carlo-Studie ``monte_carlo_paper.py`` im Unterordner ``Code/S01_Experimente/Paper/``
 und haben den Namen  (x steht für die Nummer der Studie in der Thesis)

Ergebnisse befinden sich in dem Order, der durch den Python-Path spezifiziert wurde

* 7: Daten auf lokalen Rechner übertragen
* 8: Auswertung in ``monte_carlo_paper_postprocessing.ipynb``