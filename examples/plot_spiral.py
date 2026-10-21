import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
r = np.load(sys.argv[1]); s = np.load(sys.argv[2])
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150, gridspec_kw={"width_ratios": [1, 1.25]})
g = r["g"]
a.contourf(g, g, r["Z"], levels=[-0.5, 0.5, 1.5], cmap=ListedColormap(["#d6e4fb", "#fde2cf"]))
a.contour(g, g, r["Z"], levels=[0.5], colors="#444", linewidths=0.8)
X, y = r["Xte"], r["yte"]
a.scatter(X[0, y == 0], X[1, y == 0], s=4, c="#1a73e8", label="class 0")
a.scatter(X[0, y == 1], X[1, y == 1], s=4, c="#e8710a", label="class 1")
a.set_aspect("equal"); a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_xticks([]); a.set_yticks([])
a.set_title(f"ResNet decision boundary (test acc {r['test'][-1]*100:.1f}%)", fontsize=10)
for d, name, c in [(r, "ResNet, tanh, 3 residual blocks", "#1a73e8"), (s, "Standard NN, ReLU, 20-20-20", "#9aa0a6")]:
    b.plot(np.array(d["test"]) * 100, color=c, label=f"{name}: test")
    b.plot(np.array(d["train"]) * 100, color=c, ls="--", lw=0.8, label=f"{name}: train")
b.set_xlabel("epoch"); b.set_ylabel("accuracy (%)"); b.set_ylim(40, 101); b.grid(alpha=0.3)
b.legend(fontsize=7.5, loc="lower right"); b.set_title("Training with SGD + momentum", fontsize=10)
fig.suptitle("Two-spiral classification, trained with this framework (synthetic data, 3,000 train / 1,000 test)", fontsize=10.5)
fig.tight_layout(); fig.savefig(sys.argv[3], bbox_inches="tight", facecolor="white")
