import math
import struct
from os.path import join
import sys

import numpy as np

coresuffix = ".bppr"
updtsuffix = ".updt"

print(f"The system's byte order is: {sys.byteorder}")
byteorder = '>' if (sys.byteorder == 'big') else '<'

if len(sys.argv) < 3:
    print(f"Argument list Error. Must be {sys.argv[0]} /desired/bppr/dir encodingInteger", file=sys.stderr)
    sys.exit(1) 

bpprfile = join(sys.argv[1], sys.argv[2]+coresuffix)  # Generate a training .bppr file
updtfile = join(sys.argv[1], sys.argv[2]+updtsuffix)  # Generate an OOS update/prediction .updt file
N = int(sys.argv[2]) & 0x0000FFFF                # Number of core training samples
M = (int(sys.argv[2]) & 0x007F0000) >> 16  # Number of signals generated

r = 1        # Number of ridge functions used for initialization
U = 128  # Number of OOS update samples

nu = [0]                                      # The initial degree of sigmoidized compression / rarefaction. Length r
ctrue = np.array([0.707, 0.866, 0])  # Linear weights generating y
chat = [0.9285, 0.3714, 0]       # Must be length M*r. The initial projection direction
ihat = [0.0, 0.4, 0.8, 1.0, 5.0]   # Must be length M+2. Correponds to weights [2,2,1] for signal indices=[0,1,2] respectively
ahat = [0.0, 0.33, 1.0, 3.0]       # Must be length A+2. Corresponds to weights [1,2] for nActive=[1,2] respectively
that = 0.98                                 # The initial regression-hyperparameter tau
bias = False                               # Whether an intercept term is added (not implemented)
stdX = 0.2                                  # The random noise level applied to each signal
stdY = 0.5                                  # The random noise level applied to the generated time series y. 

rng = np.random.default_rng() 
X = rng.normal(loc=0, scale=stdX, size=(N+U, M))
t = np.linspace(0, 6*np.pi, N+U);
X[:, 0] += np.sin(t)
X[:, 1] += np.cos(t)
X[:, 2] += np.tanh(np.sin(t) + np.cos(t))
y = rng.normal(loc=0, scale=0.5, size=(N+U, 1))
y += np.dot(X, ctrue[:, np.newaxis])
y += np.reshape(np.exp(-np.square(X[:, 2])) * np.sign(X[:, 2]), (N+U, 1))

with open(bpprfile, 'wb') as f:
    preFmt = f'{7}I'
    yFmt = f'{N}f'
    projFmt = f'{N*M + M*r}f'
    knotFmt = f'{r}f'
    znFmt = f'{1 + len(ihat) + len(ahat)}f'
    fmtsz = np.zeros(7, dtype=np.uint32)
    fmtsz[1:5] = np.cumsum([struct.calcsize(fmt) for fmt in [preFmt, yFmt, projFmt, knotFmt]])
    fmtsz[0] = r
    print(f"The offsets are: {fmtsz}")
    f.write(struct.pack(byteorder+preFmt, *fmtsz))
    f.write(struct.pack(byteorder+yFmt, *y[0:N].flatten('F')))
    f.write(struct.pack(byteorder+projFmt, *X[0:N, :].flatten('F'), *chat))
    f.write(struct.pack(byteorder+knotFmt, *nu))
    f.write(struct.pack(byteorder+znFmt , that, *ihat, *ahat))

with open(updtfile, 'wb') as f:
    updtFmt = f'{U*(M +1)}f'
    yX = np.concatenate((y[N:], X[N:, :]), axis=1)
    f.write(struct.pack(byteorder+updtFmt, *yX.flatten('C')))


