from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

def create_demo_data(M, N, data):
    # create some demo data for North, East, South, West
    # note that each of the 4 arrays can be either 2D (N by M) or 1D (N*M)
    # M columns and N rows

    valuesN = np.zeros(shape=(M, N), dtype=np.float64)
    valuesN[tuple(zip(*data.keys()))] = list([i.Q['↑'] for i in data.values()])
    valuesN = np.rot90(np.fliplr(valuesN))
    valuesE = np.zeros(shape=(M, N), dtype=np.float64)
    valuesE[tuple(zip(*data.keys()))] = list([i.Q['→'] for i in data.values()])
    valuesE = np.rot90(np.fliplr(valuesE))
    valuesS = np.zeros(shape=(M, N), dtype=np.float64)
    valuesS[tuple(zip(*data.keys()))] = list([i.Q['↓'] for i in data.values()])
    valuesS = np.rot90(np.fliplr(valuesS))
    valuesW = np.zeros(shape=(M, N), dtype=np.float64)
    valuesW[tuple(zip(*data.keys()))] = list([i.Q['←'] for i in data.values()])
    valuesW = np.rot90(np.fliplr(valuesW))
    return [valuesN, valuesE, valuesS, valuesW]

def triangulation_for_triheatmap(M, N):
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]
