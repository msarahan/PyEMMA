__author__ = 'noe'

import numpy as _np
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.util import types as _types

class XTRAM(_Estimator, _MultiThermModel):

    def __init__(self, lag=1, ground_state=0, count_mode='sliding', connectivity='largest',
                 dt_traj='1 step', maxiter=100000, maxerr=1e-5):
        """
        Example
        -------
        >>> from pyemma.thermo import XTRAM
        >>> import numpy as np
        >>> B = np.array([[0, 0],[0.5, 1.0]])
        >>> dtram = XTRAM(B)
        >>> traj1 = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0]]).T
        >>> traj2 = np.array([[1,1,1,1,1,1,1,1,1,1],[0,1,0,1,0,1,1,0,0,1]]).T
        >>> dtram.estimate([traj1, traj2])
        >>> dtram.log_likelihood()
        -9.8058241189353108

        >>> dtram.count_matrices
        array([[[5, 1],
                [1, 2]],

               [[1, 4],
                [3, 1]]], dtype=int32)

        >>> dtram.stationary_distribution
        array([ 0.38173596,  0.61826404])

        >>> dtram.meval('stationary_distribution')
        [array([ 0.38173596,  0.61826404]), array([ 0.50445327,  0.49554673])]

        """
        # set all parameters
        self.lag = lag
        self.ground_state = ground_state
        assert count_mode == 'sliding', 'Currently the only implemented count_mode is \'sliding\''
        self.count_mode = count_mode
        assert connectivity == 'largest', 'Currently the only implemented connectivity is \'largest\''
        self.connectivity = connectivity
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr

    def _estimate(self, trajs):
        """
        Parameters
        ----------
        trajs : ndarray(T, 2+K) or list of ndarray(T_i, 2+K)
            Thermodynamic trajectories. Each trajectory is a (T, 2+K)-array
            with T time steps. The first column is the thermodynamic state
            index, the second column is the configuration state index.
            Columns 3 to 2+K are the bias energies of the current configuation
            in all K thermodynamic states. Note that we slightly misuse float
            arrays to store integers (in columns 1 and 2), in order to have
            a compact representation of the data.

        """
        # format input if needed
        if isinstance(trajs, _np.ndarray):
            trajs = [trajs]

        # validate and reformat input
        ttrajs = []
        dtrajs = []
        Btrajs = []
        assert _types.is_list(trajs)
        ntrajs = len(trajs)
        for traj in trajs:
            _types.assert_array(traj, ndim=2, kind='numeric')
            assert _np.shape(traj)[1] > 2
            ttrajs.append(traj[:, 0].round().astype(_np.intc))
            dtrajs.append(traj[:, 1].round().astype(_np.intc))
            Btrajs.append(traj[:, 2:].astype(_np.float64))

        # set derived quantities
        import msmtools.estimation as msmest
        self.nthermo = Btrajs[0].shape[1]
        assert self.nthermo > 1, 'Only one thermodynamic state given. In this limit, TRAM is identical to ' \
                                 'a reversible Markov model. Use pyemma.msm.estimate_markov_model.'
        self.nstates_full = msmest.number_of_states(dtrajs)

        # harvest transition counts
        # TODO: replace by an efficient function. See msmtools.estimation.cmatrix
        # TODO: currently xtram_estimator only likes intc, but this should be changed
        self.count_matrices_full = _np.zeros((self.nthermo, self.nstates_full, self.nstates_full), dtype=_np.intc)
        for i in xrange(ntrajs):
            ttraj = ttrajs[i]
            dtraj = dtrajs[i]
            for t in xrange(len(ttraj)-self.lag):
                self.count_matrices_full[ttraj[t], dtraj[t], dtraj[t+self.lag]] += 1

        # connected set
        from msmtools.estimation import largest_connected_set
        Cs_flat = self.count_matrices_full.sum(axis=0)
        self.active_set = largest_connected_set(Cs_flat)

        # active set count matrices
        self.count_matrices = self.count_matrices_full[:, self.active_set, :][:, :, self.active_set]
        self.count_matrices = _np.ascontiguousarray(self.count_matrices, dtype=_np.intc)
        N = self.count_matrices.sum(axis=2)  # starting state counts
        N = _np.ascontiguousarray(N, dtype=_np.intc)

        # reduce all trajectories to connected set. We overwrite the list in order to save memory.
        in_active = _np.zeros(self.nstates_full, dtype=bool)
        in_active[self.active_set] = True
        for i in xrange(ntrajs):
            I = in_active[dtrajs[i]]
            dtrajs[i] = dtrajs[i][I]
            ttrajs[i] = ttrajs[i][I]
            Btrajs[i] = Btrajs[i][I]
        # Turn lists into matrices to collect statistics for xTRAM
        t_arr = _np.hstack(ttrajs)
        t_arr = _np.ascontiguousarray(t_arr, dtype=_np.intc)
        d_arr = _np.hstack(dtrajs)
        d_arr = _np.ascontiguousarray(d_arr, dtype=_np.intc)
        # TODO: here we transpose the B-array. We should fix a general format for all estimators.
        b_arr = _np.vstack(Btrajs).T
        b_arr = _np.ascontiguousarray(b_arr, dtype=_np.float64)

        # run estimator
        from pyemma.thermo.pytram.xtram.xtram_estimator import XTRAM as pyemma_XTRAM

        print 'nthermo', self.nthermo
        print 'nstates_full', self.nstates_full
        print 'active set ', self.active_set
        print 'counts: ', self.count_matrices
        print 'counts shape: ', self.count_matrices.shape
        print 'N: ', N
        print 'N shape: ', N.shape
        print 'B shape: ', b_arr.shape
        print 'tarr shape: ', t_arr.shape
        print 'darr shape: ', d_arr.shape

        xtram_estimator = pyemma_XTRAM(self.count_matrices, b_arr, t_arr, d_arr, N, target=self.ground_state)
        xtram_estimator.sc_iteration(maxiter=self.maxiter, ftol=self.maxerr, verbose=False)
        self.xtram_estimator = xtram_estimator

        # compute stationary distributions
        pi_all = xtram_estimator.pi_K_i
        pi_all = _np.reshape(pi_all, (xtram_estimator.n_therm_states, xtram_estimator.n_markov_states))

        # compute ML transition matrix given pi
        import msmtools.estimation as msmest
        Ps = [msmest.transition_matrix(self.count_matrices[i], reversible=True, mu=pi_all[i])
              for i in xrange(self.nthermo)]
        self.transition_matrices = _np.array(Ps)

        print 'transition matrices: ', self.transition_matrices
        print 'transition matrices shape: ', _np.shape(self.transition_matrices)

        # compute MSM objects
        for i in xrange(self.nthermo):
            self.transition_matrices[i] /= self.transition_matrices[i].sum(axis=1)[:, None]
        from pyemma.msm import MSM
        msms = [MSM(self.transition_matrices[i, :, :]) for i in xrange(self.nthermo)]
        # compute free energies of thermodynamic states and configuration states
        f_therm = xtram_estimator.f_K
        f_state = xtram_estimator.f_i
        # set model parameters to self
        self.set_model_params(models=msms, f_therm=f_therm, f=f_state)
        # done, return estimator+model
        return self


    def log_likelihood(self):
        nonzero = self.count_matrices.nonzero()
        return _np.sum(self.count_matrices[nonzero] * _np.log(self.transition_matrices[nonzero]))
