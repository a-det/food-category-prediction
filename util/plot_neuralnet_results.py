import matplotlib.pyplot as plt
import numpy as np


configs = ["Completo (10 classi)", "Top-6 classi", "Top-5 classi", "Top-5 classi + RU"]

# Reordered arrays to match: 10, 6, 5, 5 random undersampling
accuracy = [0.6065, 0.5956, 0.6385, 0.5510]
macro_f1 = [0.33, 0.47, 0.55, 0.55]
weighted_f1 = [0.58, 0.59, 0.63, 0.55]
std_cv = [0.0145, 0.0208, 0.0146, 0.0324]

x = np.arange(len(configs))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, accuracy, width, label="Accuracy")
bars2 = ax.bar(x, macro_f1, width, label="Macro F1")
bars3 = ax.bar(x + width, weighted_f1, width, label="Weighted F1")

for i, std in enumerate(std_cv):
    ax.errorbar(x[i], accuracy[i], yerr=std, fmt="o", color="black", capsize=5)

ax.set_ylabel("Score")
ax.set_title("Confronto tra configurazioni di classificazione")
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=15)
ax.set_ylim(0, 0.7)
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
