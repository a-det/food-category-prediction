import pandas as pd
import os
import re
from owlready2 import get_ontology

CSV_PATH = "data/nutrition.csv"
OWL_PATH = "ontology/foodon.owl"
SAVE_PATH = "data/nutrition_with_general_category.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Clean dataset
df = df.drop(columns=["serving_size", "Unnamed: 0", "lucopene"], errors="ignore")
df["name"] = df["name"].str.lower()
df = df.fillna(0)

# Remove units measure, keep only numbers
df = df.applymap(
    lambda x: (
        re.match(r"^\d+(\.\d+)?", str(x)).group(0)
        if isinstance(x, str) and re.match(r"^\d+(\.\d+)?", str(x))
        else x
    )
)
df = df.applymap(
    lambda x: "{:.2f}".format(float(x)) if str(x).strip() in ["0", "0.0"] else x
)

# Load ontology
onto = get_ontology(f"file://{os.path.abspath(OWL_PATH)}").load()


def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


# Iterate through all classes in the loaded ontology and, for each class,
# attempt to extract its human-readable label.
# This dictionary allows quick lookup of ontology classes
# starting from their label or name.
# Example: "tuna fillet (raw)" -> FOODON_03309012
label_to_class = {}
for classe in onto.classes():
    try:
        label = (
            classe.label.first()
            if hasattr(classe, "label") and classe.label
            else classe.name
        )
        label_to_class[label.lower()] = classe
    except Exception:
        continue


# Find the ontology class most similar to a food item from the dataset
def find_closest_class(food_name):
    if not label_to_class:
        return None

    # Find the ontology label (and thus the key in the label_to_class dictionary)
    # most similar to the food item from the dataset (i.e., food_name)
    best_label = max(
        label_to_class.keys(), key=lambda l: jaccard_similarity(food_name, l)
    )

    # Return the class (e.g., FOODON_03309012) associated with the best_label key
    return label_to_class[best_label]


# Add a column to the dataset containing the most similar ontology class
# for each food item present in the dataset
df["FoodOn_Class"] = df["name"].apply(find_closest_class)

# At this point, for each food item in the dataset we have the associated ontology class,
# but we don't yet have the general category to which the food belongs
# Example: "salad" ∈ "plant food product"


# Given an ontology class, traverse up the hierarchy of parent classes
# (using the is_a property)
def find_general_category(classe):
    try:
        if classe is None:
            return None
        # Build the path using parents from the current class
        # to reach the "food product" class
        path = []
        visited = set()
        current = classe
        while current and current not in visited:
            visited.add(current)
            labels = current.label if hasattr(current, "label") else []
            label = labels[0] if labels else current.name
            path.insert(0, (current, label))
            if label.strip().lower() == "food product":
                break
            parents = current.is_a
            if not parents:
                break
            current = parents[0]

        # Iterate through all classes in path to find the general category
        # of the food.
        found_fp = False
        for classe, label in path:
            low = label.strip().lower()
            if low == "food product":
                found_fp = True
                continue
            if found_fp:
                if "product" not in low or "by organism" not in low:
                    return label

        # If a path ending with "food product" was found
        # but none of the other classes in the path have a valid label,
        # return the immediate parent of the class for which we want
        # to find the general category.
        if found_fp and len(path) > 1:
            return path[-1][1]
    except Exception as e:
        print(f"Error finding category for {classe}: {e}")
    return None


df["General_category"] = df["FoodOn_Class"].apply(find_general_category)
df["Preferred_Label"] = df["FoodOn_Class"].apply(
    lambda x: getattr(x, "label", [x.name])[0] if x else "unknown"
)

# Remove rows without a general category
df = df.dropna(subset=["General_category"])

# Remove unnecessary columns
df = df.drop(columns=["name", "FoodOn_Class", "Preferred_Label"])

# Remove rare categories
df = df[df["General_category"].map(df["General_category"].value_counts()) >= 5]

# Save the new dataset
df.to_csv(SAVE_PATH, index=False)
print(f"Saved: {SAVE_PATH}")
