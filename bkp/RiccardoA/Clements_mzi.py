"""
Unitary matrix decomposition and circuit simulation for 6x6 universal photonic
processors using the Clements square decomposition scheme.

Created on Wed Apr  6 13:34:49 2022 @author: Ciro Pentangelo
Modified and revised on Wed Apr 25 22:51:59 2026 by Francesco Ceccarelli.

"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt

nparray = npt.NDArray


def mach_zehnder(
    m: int, n: int, internal_phase: float, external_phase: float, nmax: int
) -> npt.NDArray[np.complex128]:
    """A two-mode Mach-Zehnder interferometer section (Clements style).

    Calculated directly using the closed-form matrix representation.
    """
    mat = np.identity(nmax, dtype=np.complex128)

    g = 1j * np.exp(1j * internal_phase / 2)
    s, c = np.sin(internal_phase / 2), np.cos(internal_phase / 2)
    e_ext = np.exp(1j * external_phase)

    mat[m, m] = g * s * e_ext
    mat[m, n] = g * c
    mat[n, m] = g * c * e_ext
    mat[n, n] = -g * s
    return np.round(mat, 14)


def mach_zehnder_inv(m, n, phi_int, phi_ext, nmax):
    r"""The inverse of the Mach-Zehnder unitary matrix.
    See :func:`~.mach_zehnder` for more details on the Mach-Zehnder unitary.
    """
    return mach_zehnder(m, n, phi_int, phi_ext, nmax).conj().T


def nullMZi(m, n, U):
    r"""Nullifies element m,n of U using mach_zehnder_inv.

    Args:
        m (int): row index of element to be nullified
        n (int): column index of element to be nullified
        U (array): matrix whose m,n element is to be nullified

    Returns:
        list: list containing ``[m, n, internal_phase, external_phase, nmax]`` of the
            mach_zehnder_inv unitaries needed
    """
    nmax, mmax = U.shape

    if nmax != mmax:
        raise ValueError("U must be a square matrix")

    if U[m, n] == 0:  # no swaps for the identity-like case
        phi_i = np.pi
        phi_e = 0
    elif U[m, n + 1] == 0:  # swap in the divide-by-zero case
        phi_i = 0
        phi_e = 0
    else:
        r = -U[m, n + 1] / U[m, n]
        phi_i = 2 * np.arctan(np.abs(r))
        phi_e = -np.angle(r) % (2 * np.pi)

    return [n, n + 1, phi_i, phi_e, nmax]


def nullMZ(n, m, U):
    r"""Nullifies element n,m of U using mach_zehnder.

    Args:
        n (int): row index of element to be nullified
        m (int): column index of element to be nullified
        U (array): matrix whose m,n element is to be nullified

    Returns:
        list: list containing ``[m, n, internal_phase, external_phase, nmax]`` of the
            mach_zehnder unitaries needed
    """
    nmax, mmax = U.shape

    if nmax != mmax:
        raise ValueError("U must be a square matrix")

    if U[n, m] == 0:  # no swaps for the identity-like case
        phi_i = np.pi
        phi_e = 0
    elif U[n - 1, m] == 0:  # swap in the divide-by-zero case
        phi_i = 0
        phi_e = 0
    else:
        r = U[n - 1, m] / U[n, m]
        phi_i = 2 * np.arctan(np.abs(r))
        phi_e = -np.angle(r) % (2 * np.pi)

    return [n - 1, n, phi_i, phi_e, nmax]


def rectangular_MZ(V, tol: float = 1e-11, rearrange: bool = False):
    r"""Rectangular decomposition of a unitary matrix, with local
    phase shifts applied between two interferometers.

    Is similar to :func:`~.rectangular` except that it uses Mach Zehnder matrices to null elements of V
    using the :func:`~.null_MZ` and :func:`~.null_MZi` instead of :func:`~.T` matrices and corresponding :func:`~.nullT`
    and :func:`~.nullTi` functions.

    Args:
        V (array[complex]): unitary matrix of size n_size
        tol (float): the tolerance used when checking if the matrix is unitary

    Returns:
        tuple[array]: tuple of the form ``(tilist, np.diag(localV), tlist)``
        where:

        * ``tilist``: list containing ``[n,m,phi_int,phi_ext,n_size]`` of the ``mach_zehnder_inv`` unitaries needed
        * ``tlist``: list containing ``[n,m,phi_int,phi_ext,n_size]`` of the ``mach_zehnder`` unitaries needed
        * ``localV``: Diagonal unitary sitting sandwiched by ``mach_zehnder_inv``'s and the ``mach_zehnder``'s
    """
    localV = V
    nsize, _ = localV.shape

    if not np.allclose(V @ V.conj().T, np.identity(nsize), atol=tol, rtol=0):
        raise ValueError("The input matrix is not unitary")

    tilist = []
    tlist = []
    for k, i in enumerate(range(nsize - 2, -1, -1)):
        if k % 2 == 0:
            for j in reversed(range(nsize - 1 - i)):
                tilist.append(nullMZi(i + j + 1, j, localV))
                tilist[-1][2] %= 2 * np.pi
                tilist[-1][3] %= 2 * np.pi
                # repeat modulo operations, otherwise the input unitary
                # numpy.identity(20) yields an external_phase of exactly 2 * pi
                tilist[-1][2] %= 2 * np.pi
                tilist[-1][3] %= 2 * np.pi
                localV = localV @ mach_zehnder_inv(*tilist[-1])
        else:
            for j in range(nsize - 1 - i):
                tlist.append(nullMZ(i + j + 1, j, localV))
                tlist[-1][2] %= 2 * np.pi
                tlist[-1][3] %= 2 * np.pi
                # repeat modulo operations, otherwise the input unitary
                # numpy.identity(20) yields an external_phase of exactly 2 * pi
                tlist[-1][2] %= 2 * np.pi
                tlist[-1][3] %= 2 * np.pi
                localV = mach_zehnder(*tlist[-1]) @ localV

    for T in reversed(tlist):

        invT = mach_zehnder_inv(*T)
        localV = invT @ localV

        m, n, theta, _, nmax = T
        r = -localV[m, m] / localV[m, n]
        # theta = 2 * np.arctan(np.abs(r)) if not np.isclose(np.abs(r), 0.0) else np.pi
        phi = (
            (np.angle(r) + np.pi) % (2 * np.pi)
            if not np.isclose(np.abs(r), 0.0)
            else 0.0
        )

        invT = mach_zehnder_inv(m, n, theta, phi, nmax)
        localV = localV @ invT
        tilist.append([m, n, theta, phi, nmax])

    if rearrange:
        return localV, rearrange_clements(tilist)
    else:
        return localV, tilist


def rearrange_clements(output: list[list]) -> list[list]:
    """Reorder the list of beam splitters to match the physical mesh layout."""

    N = len(output)
    rearranged = output.copy()

    for i in range(1, N - 1):
        current_input = rearranged[i][0]
        target_input = current_input + 2

        # Max theoretical limit check based on even/odd mode parity
        max_limit = N - 1 if current_input % 2 == 1 else N - 2

        # Find the index of the next physical BS in the sequence
        if current_input < max_limit and rearranged[i + 1][0] != target_input:
            for j in range(i + 2, len(rearranged)):
                if rearranged[j][0] == target_input:
                    # Move the found element right after the current one
                    rearranged.insert(i + 1, rearranged.pop(j))
                    break
    return rearranged


# def simulate_sixmode(BS_list: list[list], rounded: bool = True) -> np.ndarray:
#     """ Simulate a 6-mode optical circuit using a list of 15 beam splitters. This
#     function works only for a six mode circuit.
#
#     Parameters
#     ----------
#     BS_list : list[list]
#         A list containing exactly 15 beam splitter configurations [ID, theta, phi].
#     rounded : bool, optional
#         If True, rounds the resulting power matrix to 6 decimal places.
#
#     Returns
#     -------
#     np.ndarray
#         A 6x6 matrix representing the power (absolute value) of the transformation.
#     """
#     assert len(BS_list) == 15
#
#     columns = []
#     m = 0
#     for i in range(6):
#         if i % 2 == 0:
#             column = []
#             for _ in range(3):
#                 column.append(BS_list[m])
#                 m += 1
#             columns.append(column.copy())
#         else:
#             column = []
#             for _ in range(2):
#                 column.append(BS_list[m])
#                 m += 1
#             columns.append(column.copy())
#     columns.reverse()
#
#     U = np.eye(6, dtype=np.complex128)
#     for column in columns:
#         blocchi = [make_T(item[1], item[2]) for item in column]
#         if len(column) == 2:
#             T = scipy.linalg.block_diag(1, blocchi[0], blocchi[1], 1)
#         else:
#             T = scipy.linalg.block_diag(*blocchi)
#         U = U @ T
#     U_power = np.abs(U)
#
#     if rounded:
#         return np.around(U_power, 6)
#     else:
#         return U_power


if __name__ == "__main__":

    from scipy.stats import unitary_group

    N = 6

    # Matrix decomposition
    np.random.seed(43)
    U_original = unitary_group.rvs(N)
    Dprime, Tlist = rectangular_MZ(U_original, rearrange=True)
    for e in Tlist: print(e)

    # Reconstruct the original U matrix
    U = Dprime
    for T in reversed(Tlist):
        U = U @ mach_zehnder(*T)
    success = np.allclose(
        U.conj().T @ U_original, np.eye(N, dtype=np.complex128), atol=1e-10
    )

    if success:
        print("SUCCESS: The matrix U has been (decomposed and) reconstruced correctly.")
    else:
        print("FAILURE: Discrepancy found between target and reconstruction.")
        max_diff = np.max(
            np.abs(U.conj().T @ U_original - np.eye(N, dtype=np.complex128))
        )
        print(f"Maximum absolute difference: {max_diff}.")
