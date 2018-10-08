import numpy as np
import matplotlib.pyplot as plt


def periodogram(t, x):
    """
    Simple periodogram for evenly sampled signals
    :param t: array-like. Sample times.
    :param x:  array-like. Signal values for thee sample times.
    :return:  p(f). Periodogram function,, to be evaluated on sample frequencies
        (usually f = 1 / t)
    """

    # Transform input data into np.array
    n = len(t)
    t = np.array(t).reshape((n, 1))
    x = np.array(x).reshape((n, 1))

    def a(t, x, f):

        f = np.array(f).reshape((1, len(f)))
        # out = 0
        # for i in range(n):
        #     out += x[i] * np.cos(2 * np.pi * f * t[i])
        out = np.sum(x * np.cos(2 * np.pi * f * t), axis=0)
        return out / n**0.5

    def b(t, x, f):

        f = np.array(f).reshape((1, len(f)))
        # out = 0
        # for i in range(n):
        #     out += x[i] * np.sin(2 * np.pi * f * t[i])
        out = np.sum(x * np.sin(2 * np.pi * f * t), axis=0)
        return out / n**0.5

    p = lambda f: a(t, x, f)**2 + b(t, x, f)**2

    return p


def ls_periodogram(t, x, centering=True, normalise=True, err=None):
    """
    Lomb-Scargle periodogram
    Computes the Lomb-Scargle periodogram (LSP) for unevenly sampled time series.
    Inputs:
        t: sample times. Array-like / iterable.
        x: values for each sample. Array-like / iterable
        centering (optional): pre-center the data by subtracting the mean value (default: True)
    Returns:
        p(f): periodogram function, to be evaluated on the desired sample frequencies.

    Example:

    import numpy as np
    import matplotlib.pyplot as plt

    n = 1000  # number of sample times
    t = 100 * np.random.rand(n)
    t.sort()  # sample times array (from 0 to 100 seconds)
    f = 0.1  # signal frequency, Hz
    x = np.sign(np.sin(2 * np.pi * f * t))  # square wave

    p = ls_periodogram(t, x)  # computes the LSP

    freqs = np.logspace(-2, 1, 2000)  # sample frequencies

    # Plot the LSP
    plt.plot(freqs, p(freqs))
    plt.xscale('log')
    plt.xlabel('$f$ [Hz]')
    plt.ylabel('Power')
    plt.show()

    --------------------------------------
    To do:
        - General code optimisation
        - Include options for periodogram normalisation, signal mean subtraction, etc
        - Implement the generalised form with floating mean (M. Zechmeister and M. Kürster, 2009)
    """

    n = len(t)

    # Transform input data into np.array
    t = np.array(t).reshape((n, 1))
    x = np.array(x).reshape((n, 1))

    if centering:
        x -= np.mean(x)

    if normalise:
        if err is not None:
            err = np.array(err).reshape((n, 1))  # column array
            W = np.sum(1 / err ** 2)
            wi = 1 / err ** 2 / W
        else:
            W = 1
            wi = 1 / n
    else:
        wi = 1
        W = n

    def a(t, x, freqs):

        freqs = np.array(freqs).reshape((1, len(freqs)))

        # cache values of tau for speed
        tau_array = tau(t, freqs)

        num = np.sum(wi * x * np.cos(2 * np.pi * freqs * (t - tau_array)), axis=0)
        den = np.sum(wi * np.cos(2 * np.pi * freqs * (t - tau_array))**2, axis=0)
        return num / den**0.5

    def b(t, x, freqs):

        freqs = np.array(freqs).reshape((1, len(freqs)))

        # cache values of tau for speed
        tau_array = tau(t, freqs)

        num = np.sum(wi * x * np.sin(2 * np.pi * freqs * (t - tau_array)), axis=0)
        den = np.sum(wi * np.sin(2 * np.pi * freqs * (t - tau_array)) ** 2, axis=0)
        return num / den**0.5

    def tau(t, f):
        """
        :param t:  column np.array
        :param f:  row np.array
        :return:  row np.array (corresponding to f)
        """

        num = np.sum(wi * np.sin(2 * np.pi * f * t), axis=0)
        den = np.sum(wi * np.cos(2 * np.pi * f * t), axis=0)
        out = np.arctan(num / den) * (4 * np.pi * f)

        return out.reshape((1, out.size))

    p = lambda f: (a(t, x, f)**2 + b(t, x, f)**2) / np.sum(wi * x**2)
    return p


def gls_periodogram(t, x, freqs, err=None, centering=False):
    """
    Generalised Lomb - Scargle Periodogram, with floating mean, measure errors and normalisation.
    Based on Zechmeister, M. and Kürster (2009)

    Inputs:
        t: sample times. Array-like / iterable.
        x: values for each sample. Array-like / iterable
        freqs: sample frequencies. Array-like / iterable.
        err (optional): measurement errors. Array-like / iterable.
    Returns:
        p(f): periodogram function, to be evaluated on the desired sample frequencies.
    """

    n = len(t)

    # Transform input data into np.array
    t = np.array(t).reshape((n, 1))  # column array
    x = np.array(x).reshape((n, 1))  # column array

    if centering:
        x -= np.mean(x)

    if err is not None:
        err = np.array(err).reshape((n, 1))  # column array
        W = np.sum(1 / err ** 2)
        wi = 1 / err ** 2 / W
    else:
        # W = 1
        wi = 1 / n

    freqs = np.array(freqs).reshape((1, len(freqs)))  # row array

    # aux variables
    y = np.sum(wi * x)
    c = np.sum(wi * np.cos(2 * np.pi * freqs * t), axis=0)
    s = np.sum(wi * np.sin(2 * np.pi * freqs * t), axis=0)
    yy = np.sum(wi * x**2) - y**2
    # yc = np.sum(wi * x * np.cos(2 * np.pi * freqs * t)) - y * c
    # ys = np.sum(wi * x * np.sin(2 * np.pi * freqs * t)) - y * s
    cc = np.sum(wi * np.cos(2 * np.pi * freqs * t)**2, axis=0) - c**2
    ss = np.sum(wi * np.sin(2 * np.pi * freqs * t)**2, axis=0) - s**2
    cs = np.sum(wi * np.cos(2 * np.pi * freqs * t) * np.sin(2 * np.pi * freqs * t), axis=0) - c * s
    tau = np.arctan2(2 * cs, (cc - ss)) / (4 * np.pi * freqs)
    yct = np.sum(wi * x * np.cos(2 * np.pi * freqs * (t - tau)), axis=0) - y * c
    yst = np.sum(wi * x * np.sin(2 * np.pi * freqs * (t - tau)), axis=0) - y * s
    cct = np.sum(wi * np.cos(2 * np.pi * freqs * (t - tau)) ** 2, axis=0) - c ** 2
    sst = np.sum(wi * np.sin(2 * np.pi * freqs * (t - tau)) ** 2, axis=0) - s ** 2

    # Periodogram
    p = 1 / yy * (yct**2 / cct + yst**2 / sst)

    return p


def plot_periodogram(t, x):

    # Filter input data
    t = np.array(t)
    x = np.array(x)

    ind = (np.invert(np.isnan(x))).nonzero()

    t = t[ind]
    x = x[ind]

    # delta_t = min(t[1:] - t[:-1])
    # tmax = max(t)
    # fmin = 1 / tmax
    # fmax = 1 / delta_t

    # start = np.log10(fmin)
    # stop = np.log10(fmax)

    # freqs = np.logspace(start, stop, num=1000)
    freqs = (1 / t)[::-1]
    freqs = freqs[:-2]

    p = periodogram(t, x)

    plt.plot(freqs, p(freqs))
    plt.xscale('log')
    plt.xlabel('$f$ [Hz]')
    # plt.ylabel('$p$ [counts$^2$s$^{-2}$]')
    plt.title('Periodogram')


def plot_ls_periodogram(t, x, fmt='-k'):

    # Filter input data
    t = np.array(t)
    x = np.array(x)

    ind = (np.invert(np.isnan(x))).nonzero()

    t = t[ind]
    x = x[ind]
    n = t.size

    delta_t = min(t[1:] - t[:-1])
    tmax = max(t)
    fmin = 1 / tmax
    fmax = 1 / delta_t

    start = np.log10(fmin)
    stop = np.log10(fmax)

    freqs = np.logspace(start, stop, num=10 * n)
    # freqs = (1 / t)[::-1]
    freqs = freqs[:-2]

    p = ls_periodogram(t, x)

    # Find local maxima
    ps = p(freqs)
    difs = np.sign(ps[1:] - ps[:-1])
    lmax = ((difs[1:] - difs[:-1]) < 0).nonzero()[0] + 1

    plt.plot(freqs, p(freqs), fmt)
    # plt.plot(freqs[lmax], ps[lmax], 'or')
    plt.xscale('log')
    plt.xlabel('$f$ [Hz]')
    # plt.ylabel('$p$ [counts$^2$s$^{-2}$]')
    plt.title('Lonb-Scargle Periodogram')
