import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def print_green(*args, **kwargs):
    GREEN = "\033[92m"
    END = "\033[0m"
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", sys.stdout)
    message = sep.join(str(arg) for arg in args)
    file.write(GREEN + message + END + end)
    file.flush()

def input_lightblue(prompt):
    LIGHTBLUE = "\033[94m"
    END = "\033[0m"
    return input(LIGHTBLUE + prompt + END)

def calculate_entropy_verbose(series, all_possible_values=None):
    """
    Berechnet die Entropie einer pandas Series und liefert zwei Strings:
      - Zeile 1: Häufigkeiten, z. B.: "Häufigkeiten: Klasse 'No' 0/4, Klasse 'Yes' 4/4"
      - Zeile 2: Formel, z. B.: "-(0/4) * log2 (0/4) - (4/4) * log2 (4/4) = 0.0000"
    Werden all_possible_values (Liste aller möglicher Klassen) übergeben, so werden auch
    Klassen mit 0 Vorkommen berücksichtigt.
    """
    counts = series.value_counts()
    total = counts.sum()
    if all_possible_values is None:
        all_possible_values = sorted(series.unique())
    fractions = []
    entropy = 0.0
    terms = []
    for value in all_possible_values:
        count = counts.get(value, 0)
        fractions.append(f"Klasse '{value}' {count}/{total}")
        prob = count / total if total > 0 else 0
        if prob > 0:
            contrib = -prob * np.log2(prob)
        else:
            contrib = 0
        entropy += contrib
        terms.append(f"({count}/{total}) * log2 ({count}/{total})")
    line1 = "Häufigkeiten: " + ", ".join(fractions)
    line2 = "-" + " - ".join(terms) + f" = {entropy:.4f}"
    return entropy, [line1, line2]

def build_id3_tree(data, attributes, target_var, print_entropy=True):
    if print_entropy:
        current_entropy, entropy_details = calculate_entropy_verbose(
            data[target_var], sorted(data[target_var].unique())
        )
        print("Gesamte Entropie der Zielvariable:")
        print(entropy_details[0])
        print(entropy_details[1])
        input_lightblue("Enter zum Fortfahren...")
    else:
        current_entropy, _ = calculate_entropy_verbose(
            data[target_var], sorted(data[target_var].unique())
        )

    # Blatt: wenn Knoten rein ist
    if abs(current_entropy) < 1e-6:
        majority = data[target_var].iloc[0]
        print("Knoten ist rein (Entropie 0). Eindeutiger Wert:", majority)
        input_lightblue("Enter zum Fortfahren...")
        return {"leaf": True, "class": majority, "num_samples": len(data)}

    # Blatt: wenn keine Attribute mehr vorhanden sind
    if not attributes:
        majority = data[target_var].mode()[0]
        print("Keine Attribute mehr vorhanden. Knoten als Blatt mit Mehrheit:", majority)
        input_lightblue("Enter zum Fortfahren...")
        return {"leaf": True, "class": majority, "num_samples": len(data)}

    total_samples = len(data)
    best_gain = -np.inf
    best_attribute = None
    best_branch_entropies = {}

    # Berechne Informationsgewinn für jedes verbleibende Attribut:
    for attribute in attributes:
        print_green("\nID3 - Berechne Gewinn für Attribut:", attribute)
        attr_weighted_entropy = 0.0
        branch_entropies = {}
        weighted_values = []  # Sammlung der gewichteten Entropieanteile als Endwerte
        for value in data[attribute].unique():
            subset = data[data[attribute] == value]
            sub_entropy, sub_details = calculate_entropy_verbose(
                subset[target_var], sorted(data[target_var].unique())
            )
            weight = len(subset) / total_samples
            weighted_component = weight * sub_entropy
            attr_weighted_entropy += weighted_component
            branch_entropies[value] = sub_entropy
            print(f"\nAttribut {attribute}, Wert {value}:")
            print(sub_details[0])
            print(sub_details[1])
            print(f"Gewicht: ({len(subset)}/{total_samples}) = {weight:.4f}")
            print(f"Gewichteter Entropieanteil: ({len(subset)}/{total_samples}) * {sub_entropy:.4f} = {weighted_component:.4f}")
            weighted_values.append(f"{weighted_component:.4f}")
        addition = " + ".join(weighted_values)
        gain = current_entropy - attr_weighted_entropy
        print(f"\nGewinn für Attribut {attribute}: {current_entropy:.4f} - ({addition}) = {gain:.4f}")
        input_lightblue("Enter zum Fortfahren...")
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
            best_branch_entropies = branch_entropies
    print(f"\nBestes Attribut gewählt: {best_attribute} (Gewinn = {best_gain:.4f})")
    input_lightblue("Enter zum Fortfahren...")

    # Für jeden Zweig des besten Attributs: Ausgabe der resultierenden Tabelle, Berechnung und Ausgabe der Entropie
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        print_green(f"\nResultierende Tabelle für {best_attribute} = {value}:\n")
        print(subset.to_string(index=False))
        branch_entropy_overall, branch_entropy_details = calculate_entropy_verbose(
            subset[target_var], sorted(data[target_var].unique())
        )
        print("\nBerechnung der Entropie für diesen Teilbaum:")
        print(branch_entropy_details[0])
        print(branch_entropy_details[1])
        print("Gesamtentropie für diesen Teilbaum: {:.4f}".format(branch_entropy_overall))
        input_lightblue("Enter zum Fortfahren...")

    # Erstelle internen Knoten:
    node = {
        "leaf": False,
        "attribute": best_attribute,
        "num_samples": total_samples,
        "branch_info": best_branch_entropies,
        "branches": {}
    }

    # Für jeden Zweig (Wert) des besten Attributs:
    for value in data[best_attribute].unique():
        print_green(f"\nID3 - Erstelle Unterbaum für {best_attribute} = {value}\n")
        subset = data[data[best_attribute] == value]
        new_attributes = [a for a in attributes if a != best_attribute]
        subtree = build_id3_tree(subset, new_attributes, target_var, print_entropy=False)
        node["branches"][value] = subtree
    return node

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Positioniert die Knoten eines Baums G hierarchisch.
    Quelle: https://stackoverflow.com/a/29597209/3954614
    """
    def _hierarchy_pos(G, root, leftmost, width, vert_gap, vert_loc, xcenter, pos, parent=None):
        pos[root] = (xcenter, vert_loc)
        neighbors = list(G.successors(root))
        if neighbors:
            dx = width / len(neighbors)
            nextx = xcenter - width/2 - dx/2
            for neighbor in neighbors:
                nextx += dx
                pos = _hierarchy_pos(G, neighbor, leftmost, dx, vert_gap, vert_loc - vert_gap, nextx, pos, root)
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_gap, vert_loc, xcenter, {})

def build_nx_tree(tree, target_var, G, parent_id=None, edge_label=None, counter=[0]):
    """
    Rekursiver Aufbau eines networkx-DiGraph aus dem Baum-Dictionary.
    - Jeder Knoten erhält eine eindeutige ID (z. B. node0, node1, ...).
    - Dem Knoten werden Labels zugewiesen, die den Attributnamen (bei inneren Knoten)
      oder den Zielvariablenwert (bei Blattknoten) sowie die Sample-Anzahl enthalten.
    - Kanten werden mit einem Label versehen, das den Klassenwert und den (optional)
      berechneten Entropiewert enthält.
    """
    current_id = f"node{counter[0]}"
    counter[0] += 1

    if tree["leaf"]:
        node_label = f"{target_var}: {tree['class']}\nSamples: {tree['num_samples']}"
    else:
        node_label = f"Attribut: {tree['attribute']}\nSamples: {tree['num_samples']}"
    G.add_node(current_id, label=node_label)

    if parent_id is not None:
        G.add_edge(parent_id, current_id, label=edge_label)

    if not tree["leaf"]:
        for branch_val, subtree in tree["branches"].items():
            branch_entropy = tree["branch_info"].get(branch_val, None)
            if branch_entropy is not None:
                child_edge_label = f"Klasse: {branch_val}\nEntropie: {branch_entropy:.4f}"
            else:
                child_edge_label = f"Klasse: {branch_val}"
            build_nx_tree(subtree, target_var, G, parent_id=current_id, edge_label=child_edge_label, counter=counter)

def draw_tree(tree, target_var):
    """
    Erzeugt eine grafische Darstellung des ID3-Baums mithilfe von networkx und matplotlib.

    Parameter:
      tree: Das Baum-Dictionary, wie es von build_id3_tree erzeugt wurde.
      target_var: Name der Zielvariablen (wird in den Knoten-Labels genutzt).

    Die Knoten werden um ca. 20% größer dargestellt, und die Kantenbeschriftungen
    werden horizontal (rotate=False) ausgegeben.
    """
    G = nx.DiGraph()
    counter = [0]
    build_nx_tree(tree, target_var, G, counter=counter)

    # Berechne ein hierarchisches Layout; der Root-Knoten hat hier die ID "node0"
    pos = hierarchy_pos(G, root="node0")

    # Extrahiere Knoten- und Kanten-Labels
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=5000, node_color='lightblue', font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=10, rotate=True)
    plt.axis('off')
    plt.show()

def main():
    try:
        print_green("ID3 - Eingabedatei laden")
        while True:
            filename = input("Bitte Quell-Datei angeben (CSV oder Excel): ").strip()

            # Prüfe, ob die Datei existiert
            if not os.path.exists(filename):
                print("Fehler: Datei nicht gefunden.\n")
                continue

            # Prüfe, ob die Datei lesbar ist
            if not os.access(filename, os.R_OK):
                print("Fehler: Datei nicht lesbar.\n")
                continue

            # Wenn Datei existiert und lesbar ist, Schleife verlassen
            break

        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext == ".csv":
            sep = input("CSV-Datei erkannt. Bitte Separator angeben (z. B. ',', ';', '|'): ").strip()
            print(f"Lade CSV-Datei mit Separator '{sep}' ...")
            df = pd.read_csv(filename, sep=sep)
        elif ext in [".xls", ".xlsx"]:
            print("Excel-Datei erkannt. Lade Excel-Datei ...")
            df = pd.read_excel(filename)
        else:
            print("Nicht unterstütztes Dateiformat!")
            return
        print("Datei wurde erfolgreich geladen!")


        print_green("\nID3 - Zielvariable auswählen")
        while True:
            print("Gefundene Variablen:")
            for idx, col in enumerate(df.columns):
                print(f"  {idx}: {col}")
            try:
                target_index = int(input("Bitte Zielvariable auswählen: "))
                target_var = df.columns[target_index]
                break  # gültige Eingabe
            except (IndexError, ValueError) as e:
                print("Ungültige Zielvariable:", e)
        print(f"Zielvariable ausgewählt: '{target_var}'")

        print_green(f"\nID3 - Berechnung der Entropie für '{target_var}'")
        input_lightblue("Enter zum Fortfahren...")
        # Hier erfolgt die Entropieberechnung NICHT in main, da sie in build_id3_tree ausgegeben wird.

        # Baum rekursiv aufbauen
        attributes = [col for col in df.columns if col != target_var]
        tree = build_id3_tree(df, attributes, target_var)

        # Baum zeichnen
        print_green("\nID3 - Zeichne Baum...\n")
        draw_tree(tree, target_var)
    except KeyboardInterrupt:
        print("\nProgramm abgebrochen.")
        sys.exit(0)

if __name__ == "__main__":
    main()
