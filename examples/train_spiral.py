import sys, json
sys.path.insert(0, sys.argv[1])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda: plt.close("all")
import numpy as np
import nn as F
from nn import NeuralNetwork
from main import F as loss_fn, getCMatrix
rng = np.random.default_rng(int(sys.argv[3]))
def spiral(n):
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    y = rng.integers(0, 2, n)
    r = t / (3 * np.pi) * 2.5
    ang = t + y * np.pi
    X = np.stack([r * np.cos(ang), r * np.sin(ang)]) + rng.normal(0, 0.08, (2, n))
    return X, y
Xtr, ytr = spiral(3000); Xte, yte = spiral(1000)
np.random.seed(int(sys.argv[3]))
arch = [int(a) for a in sys.argv[4].split(",")]; act = sys.argv[5]; res = sys.argv[6] == "1"
alpha = float(sys.argv[7]); epochs = int(sys.argv[8]); bs = 50
net = NeuralNetwork(2, arch, 2, act, ResNet=res)
hist = {"train": [], "test": [], "loss": []}
for i in range(epochs):
    net.fit(Xtr, ytr, alpha, 1, 1, bs)
    if res and i % 100 == 99: alpha *= 0.5
    hist["train"].append(float(np.mean(net.predict(Xtr) == ytr)))
    hist["loss"].append(float(loss_fn(net.x_layers[-2], net.layers_W[-1].T, getCMatrix(ytr, 2).T)))
    hist["test"].append(float(np.mean(net.predict(Xte) == yte)))
g = np.linspace(-3, 3, 300); gx, gy = np.meshgrid(g, g)
Z = net.predict(np.stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
np.savez(sys.argv[2], Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Z=Z, g=g, train=hist["train"], test=hist["test"], loss=hist["loss"], arch=str(net))
print(sys.argv[4], act, res, "final train", hist["train"][-1], "test", hist["test"][-1], "best test", max(hist["test"]))
