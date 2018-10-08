import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def get_blocks(n):
    l0 = int(np.round(np.log(n) / 2))
    blocks = []
    i = 0
    while i < n - 1:
        blocks.append([i, i + l0])
        i += l0

    if blocks[-1][1] > n - 1:
        blocks[-1][1] = n - 1

    return blocks


def neigh_blocks(d):
    """Based on Cai and Silverman (2001)"""

    n = d.size
    d_denoised = np.zeros_like(d)
    # lmbd = 4.50524  # solution to lmbd - log(lmbd) = 3
    lmbd = 0.52
    l0 = int(round(np.log(n) / 2))
    l1 = max([1, l0 / 2])
    l = int(l0 + 2 * l1)

    for i in range(n):
        s2 = np.sum(d[max(0, i - int(l / 2)):min(n, i + int(l / 2) + 1)] ** 2)
        d_denoised[i] = d[i] * max([0, (s2 - lmbd * l) / s2])

    return d_denoised


def neigh_coeffs(d):
    """Based on Cai and Silverman (2001)"""

    n = d.size
    d_denoised = np.zeros_like(d)
    lmbd = 3 / 2 * np.log(n)  # solution to lmbd - log(lmbd) = 3

    l = 3

    for i in range(d.size):
        s2 = np.sum(d[max(0, i - int(l / 2)):min(n, i + int(l / 2) + 1)] ** 2)
        k = (s2 - lmbd * l) / s2
        d_denoised[i] = d[i] * max(0, k)

    return d_denoised


def threshold(d, thr, behaviour='hard'):
    d_thr = 0
    if behaviour == 'hard':
        d_thr = d * (np.abs(d) >+ thr)
    elif behaviour == 'soft':
        d_thr = np.sign(d) * (np.abs(d) - thr) * (np.abs(d) > thr)
    return d_thr


def noise_estimator(d):
    """Using Median Absolute Deviation"""
    mad = np.median(np.abs(d - np.median(d)))
    sigma = mad / 0.6745
    return sigma


def visushrink(d, sigma=None):
    n = len(d)

    if not sigma:
        sigma = noise_estimator(d)

    thr = sigma * (2 * np.log(n)) ** 0.5

    return threshold(d, thr, behaviour='soft')


def sure_dog(d, n=2, sigma=None):
    """Based on Luisier et at (2007). SURE sum of Derivatives Of Gaussians"""

    if not sigma:
        sigma = noise_estimator(d)

    T = 6**0.5 * sigma

    def phi(y, k):
        return y * np.exp(- (k - 1) * y**2 / (2 * T**2))

    def phi_p(y, k):
        return (1 - (k - 1) * y**2 / T**2) * np.exp(- (k - 1) * y**2 / (2 * T**2))

    def m(y, n):
        mat = np.matrix([[np.mean(phi(y, i + 1) * phi(y, j + 1)) for j in range(n)] for i in range(n)])
        return mat

    def c(y, n):
        vec = np.matrix([[np.mean(y * phi(y, i + 1) - sigma**2 * phi_p(y, i + 1))] for i in range(n)])
        return vec

    try:
        a = np.linalg.solve(m(d, n), c(d, n))
    except np.linalg.LinAlgError:
        a = np.linalg.pinv(m(d, n)) * c(d, n)

    phi_mat = np.matrix([phi(d, i + 1) for i in range(n)])

    return np.array(a.T * phi_mat).ravel()


def sure_dog_interscale(d, dp, filter_size=None, n=2, sigma=None, alpha=0.5):
    """Based on Luisier et at (2007). SURE sum of Derivatives Of Gaussians"""
    if not sigma:
        sigma = noise_estimator(d)

    if not filter_size:
        filter_size = dp.size * 2 - d.size + 1
        filter_size += filter_size % 2

    # Find parent coefficients
    aux = np.array(list(zip(dp, dp))).reshape(-1)
    xp = aux[int(filter_size / 2):int(filter_size / 2) + d.size]

    T = 6**0.5 * sigma

    def phi(y, k):
        return y * np.exp(- (k - 1) * y**2 / (2 * T**2))

    def phi_p(y, k):
        return (1 - (k - 1) * y**2 / T**2) * np.exp(- (k - 1) * y**2 / (2 * T**2))

    def m(y, n):
        mat = np.matrix([[np.mean(phi(y, i + 1) * phi(y, j + 1)) for j in range(n)] for i in range(n)])
        return mat

    def c(y, n):
        vec = np.matrix([[np.mean(y * phi(y, i + 1) - sigma**2 * phi_p(y, i + 1))] for i in range(n)])
        return vec

    def f(yp):
        return np.exp(- yp**2 / (2 * T**2))

    phi_mat = np.array([phi(d, i + 1) for i in range(n)])

    try:
        a0 = np.array(np.linalg.solve(m(d, n), c(d, n)))
    except np.linalg.LinAlgError:
        a0 = np.array(np.linalg.pinv(m(d, n)) * c(d, n))

    a = a0 / ((1 - alpha) * f(xp) + alpha)
    b = alpha * a

    return np.sum((f(xp) * a + (1 - f(xp)) * b) * phi_mat, axis=0).ravel()


def denoise(x, method=sure_dog, iter=1, wavelet='sym8'):
    """
    :param x: numpy.array (1D). Array to denoise
    :param method: function to use for denoising. sure_dog, visushrink, neigh_coeffs, neigh_blocks
    :param iter: number of denoising passes
    :param wavelet: type of wavelet to use. One of pywt wavelets
    :return: denoised array, same size as original
    """

    if iter <= 1:

        # Step 1: multilevel wavelet transform
        # ------------------------------------

        coeffs = pywt.wavedec(x, wavelet, 'symmetric')

        # Approximation coefficients:
        a = coeffs[0]

        # detail coefficients
        d = coeffs[1:]

        # Step 2: denoising
        # -----------------

        # iterations

        # Noise variance estimation (for white Gaussian noise)
        # apply median absolute difference method (MAD) to the finest detail coeffs
        # (based on Donoho and Johnstone "ideal Spatial Adaptation by Wavelet Shrinkage" (1992))

        try:
            sigma = noise_estimator(d[-1])

            # denoise each detail level
            for j in range(len(d)):

                try:  # Shrink type methods
                    d[j] = method(d[j], sigma=sigma)
                except TypeError:  # Neighbour type methods
                    d[j] = method(d[j])

        except IndexError:
            pass

        # Step 3: recover the denoised time-domain sigmal
        # -----------------------------------------------

        coeffs[1:] = d
        x_denoised = pywt.waverec(coeffs, wavelet, 'symmetric')

    else:
        x_denoised = x * 1  # create a copy of x
        for i in range(iter):
            x_denoised = denoise(x_denoised, method=method, iter=1, wavelet = wavelet)

    # OUTPUT
    return x_denoised[:x.size]


def denoise_interscale(x, iter=1, wavelet='sym8', alpha=0.5):

    # Step 1: multilevel wavelet transform
    # ------------------------------------

    w = pywt.Wavelet(wavelet)
    coeffs = pywt.wavedec(x, w, 'symmetric')
    filter_size = w.dec_len

    # Approximation coefficients:
    a = coeffs[0]

    # detail coefficients
    d = coeffs[1:]

    # Step 2: denoising
    # -----------------

    # iterations
    for i in range(iter):

        # Noise variance estimation (for white Gaussian noise)
        # apply median absolute difference method (MAD) to the finest detail coeffs
        # (based on Donoho and Johnstone "ideal Spatial Adaptation by Wavelet Shrinkage" (1992))
        sigma = noise_estimator(d[-1])

        # denoise each detail level
        for j in range(len(d)):

            if j == 0:
                d[j] = sure_dog(d[j], sigma=sigma)
            else:
                d[j] = sure_dog_interscale(d[j], d[j - 1], filter_size, sigma=sigma, alpha=alpha)

    # Step 3: recover the denoised time-domain sigmal
    # -----------------------------------------------

    coeffs[1:] = d
    x_denoised = pywt.waverec(coeffs, wavelet, 'symmetric')

    # OUTPUT
    return x_denoised


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2):int(window_len / 2) + x.size]


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def noise_rejection(x, t=None, err=None):

    n = x.size

    if t is None:
        t = np.array([range(n)])

    t = t * np.std(x) / np.std(t)

    distances = (np.array([x]) - np.array([x]).T) ** 2
    distances += (t - t.T) ** 2
    distances = distances ** 0.5

    eps = 3 * np.median((np.abs(np.diff(x))) ** 2 + (np.abs(np.diff(t)) ** 2) ** 0.5)

    # distances *= np.abs(np.array([range(n)]) - np.array([range(n)]).T)

    # Make all elements outside the 3 main diagonals infinity
    # distances[np.triu_indices(n, k=2)] = 1000 * np.max(distances)
    # distances[np.tril_indices(n, k=-2)] = 1000 * np.max(distances)

    cluster = DBSCAN(eps=eps, min_samples=int(n // 100), metric='precomputed', n_jobs=8)
    cluster.fit(distances)

    ind = (cluster.labels_ >= 0).nonzero()[0]

    return ind
