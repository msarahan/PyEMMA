__author__ = 'noe'

import numpy as _np
from pyemma.util import types as _types


def metropolis(u1, u2):
    """ Metropolis function acting upon one or multiple samples

    Parameters
    ----------
    u1 : numeric, or ndarray
        energy/ies of the current configurations
    u2 : numeric, or ndarray
        energy/ies of the target configurations

    Returns
    -------
    .. math:
        M(u_1, u_2) = \min \{1, \mathrm{e}^{-(u_2-u_1)} \}

    """
    # just two numbers?
    if _types.is_float(u1) and _types.is_float(u2):
        if u1 >= u2:
            return 1.0
        else:
            return _np.exp(u1-u2)
    # handle arrays
    u1 = _types.ensure_ndarray(u1, ndim=1, kind='numeric')
    n = _np.size(u1)
    u2 = _types.ensure_ndarray(u2, ndim=1, kind='numeric', size=n)
    # default result is 1.0
    res = _np.ones(n)
    # evaluate exponential where the acceptance probability is below 1.
    I = _np.where(u1 < u2)
    res[I] = _np.exp(u1[I] - u2[I])
    return res


# TODO: rename in free_energy_bar? This function does not literally compute the bar but its log
def bar(U1, U2):
    """ Free energy differences between two thermodynamic states using Bennett's acceptance ratio (BAR).

    Estimates the free energy difference between two thermodynamic states
    using Bennett's acceptance ratio (BAR) [1]_. As an input, we need
    a set of equilibrium samples from each thermodynamic state, and for
    each sample the reduced potential energy must be evaluated at both
    thermodynamic states. Reduced potential energies are energies given
    in units of the thermal energy, often denoted by :math:`u(x) = U(x) / kT`
    where U(x) is the potential energy function and kT is the thermal
    energy.

    Parameters
    ----------
    U1 : numpy ndarray, shape=(n1, 2)
        Reduced energies for samples generated in thermodynamic state 1.
        Column 0 contains their reduced energies at thermodynamic state 1,
        column 1 contains their reduced energies at thermodynamic state 2.
    U2 : array-like, shape=(n2, 2)
        Reduced energies for samples generated in thermodynamic state 2.
        Column 0 contains their reduced energies at thermodynamic state 1,
        column 1 contains their reduced energies at thermodynamic state 2.

    Returns
    -------
    f12 : float
        free energy difference between states 1 and 2 defined by :math:`f12 = f2-f1`.

    References
    ----------
    .. [1] Bennett, C. H.: Efficient Estimation of Free Energy Differences from
        Monte Carlo Data. J. Comput. Phys. 22, 245-268 (1976)

    """
    # check input
    _types.assert_array(U1, ndim=2, kind='numeric')
    assert U1.shape[1] == 2
    _types.assert_array(U2, ndim=2, kind='numeric')
    assert U2.shape[1] == 2
    # BAR
    p12 = metropolis(U1[:, 0], U1[:, 1]).sum() / float(U1.shape[0])
    p21 = metropolis(U2[:, 1], U2[:, 0]).sum() / float(U2.shape[0])
    f12 = -_np.log(p21) + _np.log(p12)
    return f12


# TODO: think about API. This should be consistent with dTRAM, TRAM etc.
def bar_mult(ttraj, U):
    """ Relative free energies of multiple thermodynamic states using Bennett's acceptance ratio (BAR).

    Estimates free energy differences between multiple thermodynamic states by
    applying the Bennett's acceptance ratio (BAR) [1]_ to each pair of
    neighboring thermodynamic states. The result of this estimate will thus
    depend on how the thermodynamic states are ordered.

    See also
    --------
    Not to be confused with the MBAR method [2]_, [3]_ which provides a
    statistically optimal estimate for the free energy differences between
    multiple thermodynamic states. The multiple application of the two-state
    BAR method (this function) may be used in order to seed the MBAR
    optimization but will generally result in a less accurate estimate.

    Parameters
    ----------
    ttraj : int-array, size=N
        array indicating at which thermodynamic state index each sample
        was generated. Thermodynamic states are expected to be continuously
        labeled from 0 to K-1.

    U : ndarray, shape=(N, K)
        array of reduced energies of each sample evaluated at each of the K
        thermodynamic states

    Returns
    -------
    F : ndarray(K)
        relative free energies of thermodynamic states 1 through K.
        The result can be arbitrarily shifted by adding a constant.
        Here, F[0] is always set to 0 to remove that degree of freedom

    References
    ----------
    .. [1] Bennett, C. H.: Efficient Estimation of Free Energy Differences from
        Monte Carlo Data. J. Comput. Phys. 22, 245-268 (1976)

    .. [2] Bartels, C.: Chem. Phys. Lett. 331, 446 (2000).

    .. [3] Shirts, M. R. and J. D. Chodera: Statistically optimal analysis of
        samples from multiple equilibrium states.
        J. Chem. Phys. 129, 124105 (2008)

    """
    # check input
    ttraj = _types.ensure_ndarray(ttraj, ndim=1, kind='i')
    N = ttraj.size
    K = _np.max(ttraj)+1
    U = _types.ensure_ndarray(U, shape=(N, K), kind='numeric')
    # run BAR for all pairs
    dF = _np.zeros(K-1)
    for k in range(K-1):
        # pick data generated at thermodynamic state k
        I1 = _np.where(ttraj==k)[0]
        U1 = U[I1, :][:, _np.array([k, k+1])]
        I2 = _np.where(ttraj==k+1)[0]
        U2 = U[I2, :][:, _np.array([k+1, k])]
        dF[k] = bar(U1, U2)
    return _np.cumsum(dF)
