# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Array API compatible LSQR solver.

Adapted from ``scipy.sparse.linalg.lsqr``. SciPy's implementation is not
array API compatible, so the algorithm is vendored here for dense inputs.

The original Fortran code was written by C. C. Paige and M. A. Saunders as
described in:

C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
equations and sparse least squares, TOMS 8(1), 43--71 (1982).

C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
equations and least-squares problems, TOMS 8(2), 195--209 (1982).

It is licensed under the following BSD license:

Copyright (c) 2006, Systems Optimization Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

 * Neither the name of Stanford University nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The Fortran code was translated to Python for use in CVXOPT by Jeffery
Kline with contributions by Mridul Aanjaneya and Bob Myhill.

Adapted for SciPy by Stefan van der Walt.

TODO: use ``scipy.sparse.linalg.lsqr`` once it supports the array API.
"""

from math import sqrt

import numpy as np

from sklearn.utils._array_api import (
    _max_precision_float_dtype,
    get_namespace_and_device,
)

# SciPy hardcodes float64 machine epsilon for the stopping tests.
_FLOAT64_EPS = np.finfo(np.float64).eps


def _sym_ortho(a, b):
    """Stable Givens rotation, copied from ``scipy.sparse.linalg.lsqr``.

    ``a`` and ``b`` are NumPy scalars. Keeping them as ``float32`` scalars when
    the matrix is ``float32`` matches SciPy: ``math.sqrt`` is computed in
    float64 and the result is rounded back by the NumPy scalar.
    """
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


def _as_float(value):
    """Convert a Python scalar or 0-d array to ``float``."""
    return float(value)


def _numpy_dtype_for(dtype, xp):
    """NumPy dtype used for the scalar recurrence.

    SciPy's ``lsqr`` keeps bidiagonalization scalars in the dtype of ``A``.
    """
    if dtype == xp.float32:
        return np.float32
    return np.float64


def _np_scalar(value, dtype):
    """NumPy scalar of ``dtype`` holding ``value``."""
    return np.array(_as_float(value), dtype=dtype)[()]


def _mul_scalar(array, scalar, xp, device):
    """Scale ``array`` by a NumPy or Python scalar in ``array``'s dtype."""
    factor = xp.asarray(_as_float(scalar), dtype=array.dtype, device=device)
    return array * factor


def _norm(x, xp, work_dtype):
    """2-norm as a NumPy scalar of the working dtype, matching SciPy."""
    return _np_scalar(xp.linalg.vector_norm(x), work_dtype)


def _solution_dtype(xp, input_dtype, device):
    """Dtype of the solution vector.

    SciPy initializes ``x`` with ``np.zeros``, which is float64, and Ridge
    casts the result back to the input dtype afterwards. Prefer float64 when
    this namespace and device support it so the accumulated solution matches
    that behavior. Devices without float64 keep the input dtype.
    """
    if _max_precision_float_dtype(xp, device) == xp.float64:
        return xp.float64
    return input_dtype


def lsqr(
    A,
    b,
    damp=0.0,
    atol=1e-6,
    btol=1e-6,
    conlim=1e8,
    iter_lim=None,
    show=False,
    calc_var=False,
    x0=None,
    xp=None,
):
    """Find the least-squares solution to a linear system.

    Array API compatible port of ``scipy.sparse.linalg.lsqr`` for dense ``A``.
    Sparse matrices and ``LinearOperator`` inputs are not supported; callers
    that need those should use SciPy directly.

    The function solves ``Ax = b`` or ``min ||Ax - b||^2`` or
    ``min ||Ax - b||^2 + d^2 ||x - x0||^2``.

    Parameters
    ----------
    A : array of shape (m, n)
        Dense matrix.

    b : array of shape (m,)
        Right-hand side.

    damp : float, default=0.0
        Damping coefficient.

    atol, btol : float, default=1e-6
        Stopping tolerances. See ``scipy.sparse.linalg.lsqr``.

    conlim : float, default=1e8
        Stopping tolerance on the estimated condition number.

    iter_lim : int or None, default=None
        Explicit iteration limit. ``2 * n`` when None.

    show : bool, default=False
        Print an iteration log.

    calc_var : bool, default=False
        Whether to estimate diagonals of ``(A'A + damp^2 I)^{-1}``.

    x0 : array of shape (n,) or None, default=None
        Initial guess. Zeros when None.

    xp : module or None, default=None
        Array namespace. Inferred from ``A`` when omitted.

    Returns
    -------
    x : array of shape (n,)
        Solution.

    istop : int
        Reason for termination. See ``scipy.sparse.linalg.lsqr``.

    itn : int
        Iteration number upon termination.

    r1norm : float
        ``norm(b - Ax)``.

    r2norm : float
        ``sqrt(norm(r)^2 + damp^2 * norm(x - x0)^2)``.

    anorm : float
        Estimate of the Frobenius norm of ``Abar = [[A], [damp * I]]``.

    acond : float
        Estimate of ``cond(Abar)``.

    arnorm : float
        Estimate of ``norm(A' r - damp^2 (x - x0))``.

    xnorm : float
        ``norm(x)``.

    var : array of shape (n,)
        Variance estimates when ``calc_var`` is True, otherwise zeros.
    """
    xp, _, device = get_namespace_and_device(A, b, x0, xp=xp)
    if A.ndim != 2:
        raise ValueError(f"{A.ndim}-dimensional `A` is unsupported, expected 2-D.")

    b = xp.asarray(b, device=device)
    if b.ndim == 0:
        b = xp.reshape(b, (1,))
    elif b.ndim > 1:
        b = xp.squeeze(b)
    if b.ndim != 1:
        raise ValueError("b must be a one-dimensional array.")

    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(
            f"Incompatible shapes: A has {m} rows and b has {b.shape[0]} entries."
        )

    dtype = A.dtype
    # Python floats stay Python floats, matching SciPy. Array scalars (the
    # per-target ``sqrt(alpha)`` Ridge passes) keep the input dtype.
    if isinstance(damp, (float, int)):
        damp = float(damp)
    else:
        damp = _np_scalar(damp, _numpy_dtype_for(dtype, xp))
    work_dtype = _numpy_dtype_for(dtype, xp)
    atol = 1e-6 if atol is None else float(atol)
    btol = 1e-6 if btol is None else float(btol)
    conlim = 1e8 if conlim is None else float(conlim)
    if iter_lim is None:
        iter_lim = 2 * n
    else:
        iter_lim = int(iter_lim)

    if x0 is None:
        # Match SciPy: np.zeros(n) is float64 whenever that dtype exists.
        x_dtype = _solution_dtype(xp, dtype, device)
        x = xp.zeros(n, dtype=x_dtype, device=device)
    else:
        x = xp.asarray(x0, dtype=dtype, device=device, copy=True)
    var = xp.zeros(n, dtype=x.dtype, device=device)

    msg = (
        "The exact solution is x = 0 ",
        "Ax - b is small enough, given atol, btol ",
        "The least-squares solution is good enough, given atol ",
        "The estimate of cond(Abar) has exceeded conlim ",
        "Ax - b is small enough for this machine ",
        "The least-squares solution is good enough for this machine",
        "Cond(Abar) seems to be too large for this machine ",
        "The iteration limit has been reached ",
    )

    if show:
        print(" ")
        print("LSQR Least-squares solution of Ax = b")
        print(f"The matrix A has {m} rows and {n} columns")
        print(f"damp = {damp:20.14e} calc_var = {calc_var:8g}")
        print(f"atol = {atol:8.2e} conlim = {conlim:8.2e}")
        print(f"btol = {btol:8.2e} iter_lim = {iter_lim:8g}")

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy beta*u = b - A@x, alfa*v = A'@u.
    u = b
    bnorm = _norm(b, xp, work_dtype)
    if x0 is None:
        beta = bnorm
    else:
        u = u - A @ x
        beta = _norm(u, xp, work_dtype)

    if beta > 0:
        u = _mul_scalar(u, 1 / beta, xp, device)
        v = A.T @ u
        alfa = _norm(v, xp, work_dtype)
    else:
        v = xp.asarray(x, copy=True)
        alfa = 0

    w = v
    if alfa > 0:
        v = _mul_scalar(v, 1 / alfa, xp, device)
        w = xp.asarray(v, copy=True)

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0.
    arnorm = alfa * beta
    if arnorm == 0:
        if show:
            print(msg[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    if show:
        print(" ")
        print("   Itn      x[0]       r1norm     r2norm   Compatible    LS")
        test1 = 1.0
        test2 = alfa / beta
        print(
            f"{itn:6g} {_as_float(x[0]):12.5e} {r1norm:10.3e} {r2norm:10.3e}"
            f" {test1:8.1e} {test2:8.1e}"
        )

    # Main iteration loop.
    while itn < iter_lim:
        itn += 1
        # Perform the next step of the bidiagonalization to obtain the
        # next beta, u, alfa, v. These satisfy the relations
        # beta*u = A@v - alfa*u,
        # alfa*v = A'@u - beta*v.
        u = A @ v - _mul_scalar(u, alfa, xp, device)
        beta = _norm(u, xp, work_dtype)

        if beta > 0:
            u = _mul_scalar(u, 1 / beta, xp, device)
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + dampsq)
            v = A.T @ u - _mul_scalar(v, beta, xp, device)
            alfa = _norm(v, xp, work_dtype)
            if alfa > 0:
                v = _mul_scalar(v, 1 / alfa, xp, device)

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        if damp > 0:
            rhobar1 = sqrt(rhobar**2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            rhobar1 = rhobar
            psi = 0.0

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = _mul_scalar(w, 1 / rho, xp, device)

        x = x + _mul_scalar(w, t1, xp, device)
        w = v + _mul_scalar(w, t2, xp, device)
        ddnorm = ddnorm + _norm(dk, xp, work_dtype) ** 2

        if calc_var:
            var = var + dk * dk

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix Abar,
        # and the norms of rbar and Abar'rbar.
        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # Distinguish between
        # r1norm = ||b - Ax|| and
        # r2norm = sqrt(r1norm^2 + damp^2*||x - x0||^2).
        # Estimate r1norm from
        # r1norm = sqrt(r2norm^2 - damp^2*||x - x0||^2).
        # Although there is cancellation, it might be accurate enough.
        if damp > 0:
            r1sq = rnorm**2 - dampsq * xxnorm
            r1norm = sqrt(abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm

        # Use these norms to estimate other quantities, some of which will
        # be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + _FLOAT64_EPS)
        test3 = 1 / (acond + _FLOAT64_EPS)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol or ctol. The effect is equivalent to the normal tests
        # using atol = eps, btol = eps, conlim = 1/eps.
        # Later tests override earlier ones, matching SciPy.
        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if show:
            prnt = False
            if n <= 40 or itn <= 10 or itn >= iter_lim - 10:
                prnt = True
            if test3 <= 2 * ctol or test2 <= 10 * atol or test1 <= 10 * rtol:
                prnt = True
            if istop != 0:
                prnt = True
            if prnt:
                print(
                    f"{itn:6g} {_as_float(x[0]):12.5e} {r1norm:10.3e} {r2norm:10.3e}"
                    f" {test1:8.1e} {test2:8.1e} {anorm:8.1e} {acond:8.1e}"
                )

        if istop != 0:
            break

    if show:
        print(" ")
        print("LSQR finished")
        print(msg[istop])
        print(" ")
        print(
            f"istop ={istop:8g} r1norm ={r1norm:8.1e}"
            f" anorm ={anorm:8.1e} arnorm ={arnorm:8.1e}"
        )
        print(
            f"itn   ={itn:8g} r2norm ={r2norm:8.1e}"
            f" acond ={acond:8.1e} xnorm  ={xnorm:8.1e}"
        )
        print(" ")

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
