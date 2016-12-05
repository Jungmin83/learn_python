# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:33:10 2016

/// Discrete Fourier Transform without loop ///

@author: Jungmin
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1024
fs = 44100
T = N / fs
df = 1 /T
dt = 1 / fs


''' Input Signal '''
nv = np.arange(N)
kv = nv

f0 = 5000
k0 = f0 / df
x = np.cos(2 * np.pi * k0 / N * nv)


''' DFT '''
# 2D (64x1) matrix
kvv = kv[:, np.newaxis]

# For each row(each k th index), calculate s with respect to n 
# Result -> (64x64) matrix
# row : k index, column : n index
s = np.exp(1j * 2 * np.pi * kvv / N * nv)

# For each k, calculate dot product
X = np.dot(x, np.conjugate(s))


''' Plot '''
# plt.plot(kv, abs(X))
# plt.axis([0, N-1, 0, N])

f = np.arange(0, fs, df)
plt.plot(f, abs(X))
plt.axis([0, fs-df, 0, N])

plt.xlabel('frequency index')
plt.ylabel('amplitude')

plt.show()