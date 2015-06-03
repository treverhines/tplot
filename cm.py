#!/usr/bin/env python
import matplotlib.colors

_slip = [[1.00000,1.00000,1.00000],
[0.98824,1.00000,1.00000],
[0.97255,1.00000,1.00000],
[0.95686,1.00000,1.00000],
[0.94118,0.99608,1.00000],
[0.92549,0.99608,1.00000],
[0.90980,0.99608,1.00000],
[0.89412,0.99216,1.00000],
[0.87843,0.99216,1.00000],
[0.86275,0.99216,1.00000],
[0.84706,0.98824,1.00000],
[0.83137,0.98824,1.00000],
[0.81569,0.98431,1.00000],
[0.80000,0.98431,1.00000],
[0.78431,0.98431,1.00000],
[0.76863,0.98039,1.00000],
[0.75294,0.98039,1.00000],
[0.73725,0.98039,1.00000],
[0.72157,0.97647,1.00000],
[0.70588,0.97647,1.00000],
[0.69020,0.97647,1.00000],
[0.67451,0.97255,1.00000],
[0.65882,0.97255,1.00000],
[0.63922,0.96863,1.00000],
[0.62353,0.96863,1.00000],
[0.60784,0.96863,1.00000],
[0.59216,0.96471,1.00000],
[0.57647,0.96471,1.00000],
[0.56078,0.96471,1.00000],
[0.54510,0.96078,1.00000],
[0.52941,0.96078,1.00000],
[0.51373,0.96078,1.00000],
[0.49804,0.95686,1.00000],
[0.48235,0.95686,1.00000],
[0.46667,0.95294,1.00000],
[0.45098,0.95294,1.00000],
[0.43529,0.95294,1.00000],
[0.41961,0.94902,1.00000],
[0.40392,0.94902,1.00000],
[0.38824,0.94902,1.00000],
[0.37255,0.94510,1.00000],
[0.35686,0.94510,1.00000],
[0.34118,0.94510,1.00000],
[0.32549,0.94118,1.00000],
[0.30980,0.94118,1.00000],
[0.29412,0.93725,1.00000],
[0.27843,0.93725,1.00000],
[0.26275,0.93725,1.00000],
[0.24706,0.93333,1.00000],
[0.23137,0.93333,1.00000],
[0.21569,0.93333,1.00000],
[0.20000,0.92941,1.00000],
[0.18431,0.92941,1.00000],
[0.16863,0.92941,1.00000],
[0.15294,0.92549,1.00000],
[0.13725,0.92549,1.00000],
[0.12157,0.92157,1.00000],
[0.10588,0.92157,1.00000],
[0.09020,0.92157,1.00000],
[0.07451,0.91765,1.00000],
[0.05882,0.91765,1.00000],
[0.04314,0.91765,1.00000],
[0.02745,0.91373,1.00000],
[0.01176,0.91373,1.00000],
[0.00392,0.91373,1.00000],
[0.01569,0.91373,0.98431],
[0.03137,0.91373,0.96863],
[0.04706,0.91765,0.95294],
[0.05882,0.91765,0.93725],
[0.07451,0.92157,0.92157],
[0.09020,0.92157,0.90588],
[0.10196,0.92157,0.89020],
[0.11765,0.92549,0.87451],
[0.12941,0.92549,0.85882],
[0.14510,0.92549,0.84314],
[0.15686,0.92941,0.82745],
[0.17255,0.92941,0.81176],
[0.18824,0.92941,0.79608],
[0.20000,0.93333,0.78039],
[0.21569,0.93333,0.76471],
[0.22745,0.93725,0.74902],
[0.24314,0.93725,0.73333],
[0.25882,0.93725,0.71765],
[0.27059,0.94118,0.70196],
[0.28627,0.94118,0.68627],
[0.29804,0.94118,0.67059],
[0.31373,0.94510,0.65490],
[0.32941,0.94510,0.63529],
[0.34118,0.94510,0.61961],
[0.35686,0.94902,0.60392],
[0.37255,0.94902,0.58824],
[0.38431,0.95294,0.57255],
[0.40000,0.95294,0.55686],
[0.41176,0.95294,0.54118],
[0.42745,0.95686,0.52549],
[0.44314,0.95686,0.50980],
[0.45490,0.95686,0.49412],
[0.47059,0.96078,0.47843],
[0.48235,0.96078,0.46275],
[0.49804,0.96078,0.44706],
[0.51373,0.96471,0.43137],
[0.52549,0.96471,0.41569],
[0.54118,0.96863,0.40000],
[0.55294,0.96863,0.38431],
[0.56863,0.96863,0.36863],
[0.58039,0.97255,0.35294],
[0.59608,0.97255,0.33725],
[0.61176,0.97255,0.32157],
[0.62353,0.97647,0.30588],
[0.63922,0.97647,0.29020],
[0.65490,0.97647,0.27451],
[0.66667,0.98039,0.25882],
[0.68235,0.98039,0.24314],
[0.69412,0.98431,0.22745],
[0.70980,0.98431,0.21176],
[0.72549,0.98431,0.19608],
[0.73725,0.98824,0.18039],
[0.75294,0.98824,0.16471],
[0.76471,0.98824,0.14902],
[0.78039,0.99216,0.13333],
[0.79608,0.99216,0.11765],
[0.80784,0.99216,0.10196],
[0.82353,0.99608,0.08627],
[0.83529,0.99608,0.07059],
[0.85098,1.00000,0.05490],
[0.86667,1.00000,0.03922],
[0.87843,1.00000,0.02353],
[0.89412,1.00000,0.00784],
[0.90196,1.00000,0.00000],
[0.90196,0.99216,0.00000],
[0.90588,0.98431,0.00000],
[0.90588,0.97647,0.00000],
[0.90588,0.96471,0.00000],
[0.90980,0.95686,0.00000],
[0.90980,0.94902,0.00000],
[0.91373,0.94118,0.00000],
[0.91373,0.93333,0.00000],
[0.91373,0.92549,0.00000],
[0.91765,0.91765,0.00000],
[0.91765,0.90980,0.00000],
[0.92157,0.90196,0.00000],
[0.92157,0.89412,0.00000],
[0.92157,0.88627,0.00000],
[0.92549,0.87843,0.00000],
[0.92549,0.87059,0.00000],
[0.92941,0.86275,0.00000],
[0.92941,0.85098,0.00000],
[0.93333,0.84314,0.00000],
[0.93333,0.83529,0.00000],
[0.93333,0.82745,0.00000],
[0.93725,0.81961,0.00000],
[0.93725,0.81176,0.00000],
[0.94118,0.80392,0.00000],
[0.94118,0.79608,0.00000],
[0.94118,0.78824,0.00000],
[0.94510,0.78039,0.00000],
[0.94510,0.77255,0.00000],
[0.94902,0.76471,0.00000],
[0.94902,0.75686,0.00000],
[0.94902,0.74510,0.00000],
[0.95294,0.73725,0.00000],
[0.95294,0.72941,0.00000],
[0.95686,0.72157,0.00000],
[0.95686,0.71373,0.00000],
[0.96078,0.70588,0.00000],
[0.96078,0.69804,0.00000],
[0.96078,0.69020,0.00000],
[0.96471,0.68235,0.00000],
[0.96471,0.67451,0.00000],
[0.96863,0.66667,0.00000],
[0.96863,0.65882,0.00000],
[0.96863,0.65098,0.00000],
[0.97255,0.64314,0.00000],
[0.97255,0.63137,0.00000],
[0.97647,0.62353,0.00000],
[0.97647,0.61569,0.00000],
[0.97647,0.60784,0.00000],
[0.98039,0.60000,0.00000],
[0.98039,0.59216,0.00000],
[0.98431,0.58431,0.00000],
[0.98431,0.57647,0.00000],
[0.98824,0.56863,0.00000],
[0.98824,0.56078,0.00000],
[0.98824,0.55294,0.00000],
[0.99216,0.54510,0.00000],
[0.99216,0.53725,0.00000],
[0.99608,0.52941,0.00000],
[0.99608,0.51765,0.00000],
[0.99608,0.50980,0.00000],
[1.00000,0.50196,0.00000],
[1.00000,0.49412,0.00000],
[1.00000,0.48627,0.00000],
[0.99608,0.47843,0.00000],
[0.98824,0.47059,0.00000],
[0.98039,0.46275,0.00000],
[0.97255,0.45490,0.00000],
[0.96471,0.44706,0.00000],
[0.95686,0.43922,0.00000],
[0.95294,0.43529,0.00000],
[0.94510,0.42745,0.00000],
[0.93725,0.41961,0.00000],
[0.92941,0.41176,0.00000],
[0.92157,0.40392,0.00000],
[0.91373,0.39608,0.00000],
[0.90588,0.38824,0.00000],
[0.89804,0.38039,0.00000],
[0.89020,0.37255,0.00000],
[0.88235,0.36471,0.00000],
[0.87451,0.35686,0.00000],
[0.86667,0.34902,0.00000],
[0.85882,0.34118,0.00000],
[0.85098,0.33333,0.00000],
[0.84314,0.32549,0.00000],
[0.83529,0.31765,0.00000],
[0.82745,0.31373,0.00000],
[0.81961,0.30588,0.00000],
[0.81176,0.29804,0.00000],
[0.80392,0.29020,0.00000],
[0.79608,0.28235,0.00000],
[0.78824,0.27451,0.00000],
[0.78039,0.26667,0.00000],
[0.77255,0.25882,0.00000],
[0.76471,0.25098,0.00000],
[0.75686,0.24314,0.00000],
[0.74902,0.23529,0.00000],
[0.74510,0.22745,0.00000],
[0.73725,0.21961,0.00000],
[0.72941,0.21176,0.00000],
[0.72157,0.20392,0.00000],
[0.71373,0.19608,0.00000],
[0.70588,0.18824,0.00000],
[0.69804,0.18431,0.00000],
[0.69020,0.17647,0.00000],
[0.68235,0.16863,0.00000],
[0.67451,0.16078,0.00000],
[0.66667,0.15294,0.00000],
[0.65882,0.14510,0.00000],
[0.65098,0.13725,0.00000],
[0.64314,0.12941,0.00000],
[0.63529,0.12157,0.00000],
[0.62745,0.11373,0.00000],
[0.61961,0.10588,0.00000],
[0.61176,0.09804,0.00000],
[0.60392,0.09020,0.00000],
[0.59608,0.08235,0.00000],
[0.58824,0.07451,0.00000],
[0.58039,0.06667,0.00000],
[0.57255,0.06275,0.00000],
[0.56471,0.05490,0.00000],
[0.55686,0.04706,0.00000],
[0.54902,0.03922,0.00000],
[0.54118,0.03137,0.00000],
[0.53333,0.02353,0.00000],
[0.52941,0.01569,0.00000],
[0.52157,0.00784,0.00000],
[0.51373,0.00000,0.00000]]

slip = matplotlib.colors.ListedColormap(_slip)
slip_r = matplotlib.colors.ListedColormap(_slip[::-1])