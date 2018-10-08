import numpy as np


class BayesBlocks (object):

    """Creates Bayesian blocks from input data.
    Based on Scargle et al, 2013

    :param data: dict containing one or more of the following keys:
        - 't': iterable (1D) containing the times of events or bins.
        - 'x': iterable (1D) containing the values for each bin or events, in case of time series
            with errors or binned data.
        - 'err': errors for time series with errors. Iterable (1D)
    :param options: (optional). Dict containing some of the following keys, if necessary:
        - 'p0':= 0.05. False - positive rate for the prior on the number of points.
        - 'iterate':= False. flag that indicates if iteration in p0 is required.
        - 'max_iter':= 20. maximum number of iterations.
        - 'relax':= 1. Relaxation factor for the iterations in p0.
    :return object: with attributes data, options, blocks. obj.blocks is a dict containing the following keys:
        - 'change_points': array if indexes of the change points.
        - 'bins': list of tuples. Each tuple contains the start and end times for each block.
        - 'x:blocks': array of mean values for each block.
    """

    # Default options dictionary
    options = {
        'p0': 0.05,  # false - positive rate
        'iterate': False,  # iterations in false - positive rate
        'max_iter': 20,  # maximum number of iterations
        'relax': 1  # relaxation factor for the iterations in p0
    }
    blocks = {}

    def __init__(self, data, options=None):
        self.data = {}  # data dictionary. keys 't', 'x', 'err'
        if 'err' in data.keys():
            self.options['mode'] = 3  # Time series with errors
            self.data['err'] = np.array(data['err'])
            try:
                self.data['x'] = np.array(data['x'])
                self.data['n'] = self.data['x'].size
            except KeyError:
                raise KeyError('Data point values missing. Please include key x in data.')
            try:
                self.data['t'] = np.array(data['t'])
            except KeyError:
                self.data['t'] = np.arange(self.data['n'])

        elif 'x' in data.keys():
            self.options['mode'] = 2  # binned data
            self.data['x'] = np.array(data['x'])
            self.data['n'] = self.data['x'].size
            try:
                self.data['t'] = np.array(data['t'])
            except KeyError:
                self.data['t'] = np.arange(self.data['n'])

        else:
            self.options['mode'] = 1  # Timed events
            try:
                self.data['t'] = np.array(data['t'])
                self.data['n'] = self.data['t'].size
            except KeyError:
                raise KeyError('Event times must be provided')
            self.data['x'] = np.ones(self.data['n'])

        # Override default options if new ones provided
        if options:
            for o in options.keys():
                if o in self.options.keys():
                    self.options[o] = options[o]

        # Get the change points
        self.blocks['change_points'] = self.get_change_points()

        # Iterate in p0 if required in options
        if self.options['iterate']:
            self.iterate_p0()

        # Get the values for block average and bin times
        self.get_block_coordinates()

    def get_change_points(self):
        x = self.data['x']
        t = self.data['t']
        n = self.data['n']
        ncp_prior = self.prior_fcn()

        if self.options['mode'] == 3:
            err = self.data['err']
        else:
            edges = np.concatenate((
                t[:1],
                0.5 * (t[1:] + t[:-1]),
                t[-1:]
            ))
            b_length = t[-1] - edges
            self.data['edges'] = edges

        best = np.zeros(n, dtype=float)
        last = np.zeros(n, dtype=int)

        for k in range(n):
            if self.options['mode'] == 3:
                a = 0.5 * np.cumsum(1 / err[k::-1]**2)
                b = -np.cumsum(x[k::-1] / err[k::-1]**2)
                fit_vec = b[::-1]**2 / (4 * a[::-1])

            else:
                arg_log = b_length[:k + 1] - b_length[k + 1]
                ind = (arg_log <= 0).nonzero()
                arg_log[ind] = np.inf
                count_vec = np.cumsum(x[:k + 1][::-1])[::-1]

                fit_vec = count_vec * (np.log(count_vec) - np.log(arg_log))

            aux = fit_vec + np.concatenate(([0], best[:k])) - ncp_prior
            ind = np.argmax(aux)
            best[k] = aux[ind]
            last[k] = ind

        change_points = np.array([], dtype=int)
        ind = last[-1]
        while ind > 0:
            change_points = np.concatenate(([ind], change_points))
            ind = last[ind - 1]

        return change_points

    def iterate_p0(self):
        old = None
        it_count = 0
        while it_count <= self.options['max_iter']:
            it_count += 1
            chpoints = self.blocks['change_points']
            n_cp = chpoints.size
            if n_cp < 1:
                n_cp = 1
            # if 'old' in locals():
            try:
                if old.size == n_cp:
                    it_err = sum(np.abs(chpoints - old))
                else:
                    it_err = np.inf

                if it_err == 0:
                    print('Prior iteration converged in', it_count, 'iterations, with p0 = ',
                          self.options['p0'])
                    break
                elif it_count == self.options['max_iter']:
                    print('Did not converge in', it_count, 'iterations')
                    break
            except AttributeError:  # executed if 'old' has not been called yet (first iteration).
                pass

            self.options['p0'] = 1 - (1 - self.options['p0']) / (n_cp * self.options['relax'])
            old = chpoints
            self.blocks['change_points'] = self.get_change_points()

    def prior_fcn(self):
        p0 = self.options['p0']
        n = self.data['n']

        return 4 - np.log(p0 / (0.0163 * n**0.478))

    def get_block_coordinates(self):
        chpoints = self.blocks['change_points']
        t = self.data['t']
        x = self.data['x']
        ind_array = np.concatenate(([0], chpoints, [t.size - 1]))

        x_mean = []
        bins = []

        for i in range(ind_array.size - 1):
            i0 = ind_array[i]
            i1 = ind_array[i + 1]
            if i == ind_array.size - 2:
                x_slice = x[i0:]
            else:
                x_slice = x[i0:i1]
            x_mean = np.concatenate((x_mean, [np.mean(x_slice)]))
            bins += [(t[i0], t[i1])]

        self.blocks['x_blocks'] = x_mean
        self.blocks['bins'] = bins
