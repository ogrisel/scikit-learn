# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import mlx.core as mx
import numpy as np

from sklearn.externals.array_api_compat import mlx as xp

mx.set_default_device(mx.cpu)
rng = np.random.default_rng(0)
A = rng.normal(size=(100, 20))
C = A.T @ A / 100
Cm = xp.asarray(C)
Am = xp.asarray(A)


def rel(a, b):
    return np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b))


print("pinv", rel(xp.linalg.pinv(Cm), np.linalg.pinv(C)))
print("inv", rel(xp.linalg.inv(Cm), np.linalg.inv(C)))
print("svdvals", rel(xp.linalg.svdvals(Cm), np.linalg.svd(C, compute_uv=False)))
U, S, Vt = mx.linalg.svd(Cm, stream=mx.cpu)
print("raw svd recon", rel(U[:, :20] * S @ Vt, C))
U, S, Vt = xp.linalg.svd(Am, full_matrices=False)
print("thin svd recon", rel((U * S) @ Vt, A))
Q, R = xp.linalg.qr(Am)
print("qr recon", rel(Q @ R, A))
w, V = xp.linalg.eigh(Cm)
print("eigh vals", rel(w, np.linalg.eigvalsh(C)))
L = xp.linalg.cholesky(Cm)
print("cholesky", rel(L, np.linalg.cholesky(C)))
print("solve", rel(xp.linalg.solve(Cm, Cm[:, :3]), np.linalg.solve(C, C[:, :3])))
