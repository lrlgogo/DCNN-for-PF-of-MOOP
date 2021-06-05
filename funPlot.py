import math as mt
import matplotlib.pyplot as plt

def funSchafferN1(x):
    return [x**2, (x - 2)**2]

def funPlot1D(f, xScale, yDim=2, xRes=10000):
    y = [[] for i in range(yDim)]
    x = []
    xMin, xMax = xScale
    for xp in range(xRes):
        x.append(xMin + (xMax - xMin) * xp / (xRes - 1))
        yL = f(x[-1])
        for i in range(yDim):
            y[i].append(yL[i])
    plt.plot(y[0], y[1], 'b', label='Test')
    plt.show()

funPlot1D(funSchafferN1, [-10, 10])
