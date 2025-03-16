import os
import sys
import numpy as np
import pandas as pd

def print_green(*args, **kwargs):
    GREEN = "\033[92m"
    END = "\033[0m"
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    message = sep.join(str(arg) for arg in args)
    sys.stdout.write(GREEN + message + END + end)
    sys.stdout.flush()

def input_lightblue(prompt):
    LIGHTBLUE = "\033[94m"
    END = "\033[0m"
    return input(LIGHTBLUE + prompt + END)

def select_file():
    while True:
        print_green("Naive Bayes - Eingabedatei laden\n")
        filename = input("Bitte Quell-Datei angeben (CSV oder Excel): ").strip()
        if not os.path.exists(filename):
            print("Fehler: Datei nicht gefunden.\n")
            continue
        if not os.access(filename, os.R_OK):
            print("Fehler: Datei nicht lesbar.\n")
            continue
        return filename

def select_target_variable(df):
    print_green("\nNaive Bayes - Zielvariable auswählen\n")
    while True:
        print("Gefundene Variablen:")
        for idx, col in enumerate(df.columns):
            print(f"  {idx}: {col}")
        try:
            target_index = int(input("Bitte Zielvariable auswählen: "))
            target_var = df.columns[target_index]
            return target_var
        except (IndexError, ValueError) as e:
            print("Ungültige Eingabe für die Zielvariable:", e)
            print("Bitte erneut versuchen.\n")

def print_relative_frequencies(df, target_var):
    """
    Gibt für alle Attribute (außer der Zielvariable) eine kompakte Tabelle aus.
    Der Header zeigt z. B. "Play: yes(3) | no(5)" – also die Gesamtanzahl je Zielklasse.
    Für jedes Attribut wird für jeden Attributswert eine Zeile ausgegeben, z. B.:
      Outlook(sunny)    | 0/3 | 3/5
      Outlook(overcast) | 3/3 | 2/5
      Temp(high)        | 1/3 | 1/5
      Temp(medium)      | 1/3 | 2/5
      Temp(cold)        | 1/3 | 2/5
    """
    classes = sorted(df[target_var].unique())
    total_counts = {cls: df[df[target_var] == cls].shape[0] for cls in classes}
    header = f"{target_var}: " + " | ".join(f"{cls}({total_counts[cls]})" for cls in classes)
    print_green("\nRelative Häufigkeiten für alle Attribute:\n")
    print(header)
    print("-" * len(header))
    for attribute in df.columns:
        if attribute == target_var:
            continue
        unique_values = sorted(df[attribute].unique())
        for val in unique_values:
            row_label = f"{attribute}({val})"
            row_values = []
            for cls in classes:
                count = df[(df[attribute] == val) & (df[target_var] == cls)].shape[0]
                row_values.append(f"{count}/{total_counts[cls]}")
            print(f"{row_label:<20} | " + " | ".join(row_values))

def select_new_sample(df, target_var):
    sample = {}
    print_green("\nNeue Fallwerte eingeben (für jedes Attribut)")
    for attribute in df.columns:
        if attribute == target_var:
            continue
        unique_values = sorted(df[attribute].unique())
        print(f"\nAttribut: {attribute}")
        for idx, val in enumerate(unique_values):
            print(f"  {idx}: {val}")
        while True:
            try:
                idx_val = int(input(f"Wählen Index für {attribute}: "))
                sample[attribute] = unique_values[idx_val]
                break
            except (IndexError, ValueError) as e:
                print("Ungültige Eingabe...")
    return sample

def compute_likelihoods_and_posteriors(df, target_var, sample):
    """
    Berechnet für jede Zielklasse die Likelihood als Produkt der bedingten Wahrscheinlichkeiten.
    """
    classes = sorted(df[target_var].unique())
    likelihoods = {}
    details = {}
    total_samples = len(df)
    print_green("\nBerechnung der Likelihoods:\n")
    for cls in classes:
        total_cls = df[df[target_var] == cls].shape[0]
        likelihood = 1.0
        cls_details = []
        for attribute, value in sample.items():
            count = df[(df[attribute] == value) & (df[target_var] == cls)].shape[0]
            frac = f"{count}/{total_cls}"
            cls_details.append(frac)
            likelihood *= (count / total_cls) if total_cls > 0 else 0
        # Prior: relative Häufigkeit der Klasse (ohne Laplace)
        prior_frac = f"{total_cls}/{total_samples}"
        cls_details.append(prior_frac)
        likelihood *= (total_cls / total_samples)
        details[cls] = cls_details
        detail_str = " * ".join(cls_details)
        print(f"Likelihood: {target_var}({cls}) = {detail_str} = {likelihood:.4f}")
        likelihoods[cls] = likelihood
    return likelihoods, details

def compute_likelihoods_and_posteriors_laplace(df, target_var, sample, smoothing=1):
    """
    Berechnet die Likelihoods mit LaPlace-Korrektur.
    """
    classes = sorted(df[target_var].unique())
    likelihoods = {}
    details = {}
    total_samples = len(df)
    num_classes = len(classes)
    print_green("\nBerechnung der Likelihoods mit LaPlace:\n")
    # Ausgabe für jedes Attribut im Sample:
    for attribute in sample.keys():
        V = df[attribute].nunique()
        print_green(f"{attribute} - {V} Ausprägungen - addiere 1/{V}")
    print("\n")
    for cls in classes:
        total_cls = df[df[target_var] == cls].shape[0]
        cls_details = []
        for attribute, value in sample.items():
            V = df[attribute].nunique()
            count = df[(df[attribute] == value) & (df[target_var] == cls)].shape[0]
            frac_str = f"({count}+1)/({total_cls}+{V})"
            cls_details.append(frac_str)
        # Prior
        prior_frac = f"{total_cls}/{total_samples}"
        cls_details.append(prior_frac)
        details[cls] = cls_details
        # Berechne den tatsächlichen Wert:
        likelihood = 1.0
        for attribute, value in sample.items():
            V = df[attribute].nunique()
            count = df[(df[attribute] == value) & (df[target_var] == cls)].shape[0]
            prob = (count + smoothing) / (total_cls + smoothing * V) if total_cls > 0 else 0
            likelihood *= prob
        prior = total_cls / total_samples
        likelihood *= prior
        likelihoods[cls] = likelihood
        detail_str = " * ".join(cls_details)
        print(f"Likelihood: {target_var}({cls}) mit LaPlace: {detail_str} = {likelihood:.4f}")
    return likelihoods, details

def compute_normalized_posteriors(likelihoods, target_var):
    total = sum(likelihoods.values())
    normalized = {}
    print_green("\nBerechnung der normalisierten Wahrscheinlichkeiten:\n")
    parts = " + ".join(f"{likelihoods[cls]:.4f}" for cls in sorted(likelihoods.keys()))
    for cls in sorted(likelihoods.keys()):
        norm = likelihoods[cls] / total if total != 0 else 0
        normalized[cls] = norm
    for cls in sorted(likelihoods.keys()):
        if total == 0:
            print(f"Wahrscheinlichkeit: {target_var}({cls}) = 0 (keine Wahrscheinlichkeit berechenbar)")
        else:
            print(f"Wahrscheinlichkeit: {target_var}({cls}) = {likelihoods[cls]:.4f} / ({parts}) = {normalized[cls]:.2f} entspricht {normalized[cls]*100:.0f}%")
    return normalized

def main():
    try:
        # Schritt 1: Quelldatei laden
        filename = select_file()
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext == ".csv":
            sep = input("CSV-Datei erkannt. Bitte geben Sie den Separator ein (z. B. ',', ';', '|'): ").strip()
            print(f"Lade CSV-Datei mit Separator '{sep}' ...")
            df = pd.read_csv(filename, sep=sep)
        elif ext in [".xls", ".xlsx"]:
            print("Excel-Datei erkannt. Lade Excel-Datei ...")
            df = pd.read_excel(filename)
        else:
            print("Nicht unterstütztes Dateiformat!")
            return
        print("Datei wurde erfolgreich geladen!")

        # Schritt 2: Zielvariable auswählen
        target_var = select_target_variable(df)
        print(f"Zielvariable ausgewählt: '{target_var}'")

        # Schritt 3: Tabelle der relativen Häufigkeiten ausgeben
        print_relative_frequencies(df, target_var)

        # Schritt 4: Neue Fallwerte eingeben
        sample = select_new_sample(df, target_var)
        print_green("\nGewählte Fallwerte:\n")
        for attr, val in sample.items():
            print(f"{attr}: {val}")

        # Schritt 5: Likelihoods berechnen
        likelihoods, details = compute_likelihoods_and_posteriors(df, target_var, sample)

        # Schritt 6: Normalisierte Wahrscheinlichkeiten berechnen
        posteriors = compute_normalized_posteriors(likelihoods, target_var)

        print_green("\nErgebnis:\n")
        for cls in sorted(posteriors.keys()):
            print(f"{target_var}({cls}): {posteriors[cls]*100:.0f}%")

        # Schritt 7: Falls 0% oder 100% auftritt, Laplace-Korrektur anwenden
        if any(p == 0 or p == 1 for p in posteriors.values()):
            laplace_likelihoods, laplace_details = compute_likelihoods_and_posteriors_laplace(df, target_var, sample)
            laplace_posteriors = compute_normalized_posteriors(laplace_likelihoods, target_var)
            print_green("\nErgebnis mit LaPlace-Korrektur:\n")
            for cls in sorted(laplace_posteriors.keys()):
                print(f"{target_var}({cls}): {laplace_posteriors[cls]*100:.0f}%")
            print("\n")

    except KeyboardInterrupt:
        print("\nProgramm abgebrochen.")
        sys.exit(0)

if __name__ == "__main__":
    main()
