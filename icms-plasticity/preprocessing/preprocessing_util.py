import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def simple(data):
    """
    Detrend signal simply by subtracting a line through the first and last
    point of the trace

    :param data: Data to detrend, type numpy.ndarray.
    :return: Detrended data. Returns the original array which has been
        modified in-place if possible but it might have to return a copy in
        case the dtype has to be changed.
    """
    # Convert data if it's not a floating point type.
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    data -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
    return data


def _plotting_helper(data, fits, plot):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    plt.subplots_adjust(hspace=0)

    # Plot original data and fits for each channel
    for i in range(data.shape[1]):
        axes[0].plot(data[:, i], label=f"Channel {i+1} Original", color='k')
        axes[0].plot(fits[i], label=f"Channel {i+1} Fit", lw=1, color='r')

    # axes[0].legend(loc="best")
    axes[0].label_outer()

    # Plot residuals for each channel
    for i in range(data.shape[1]):
        axes[1].plot(data[:, i] - fits[i],
                     label=f"Channel {i+1} Residual", color='k')

    # axes[1].legend(loc="best")
    axes[1].label_outer()
    axes[1].set_xlabel("Samples")

    plt.tight_layout(h_pad=0)

    if plot is True:
        plt.show()
    else:
        plt.savefig(plot)
        plt.close(fig)


def polynomial(data, order, plot=False):
    """
    Removes a polynomial trend from the data.

    :param data: The data to detrend. Will be modified in-place.
    :type data: :class:`numpy.ndarray`
    :param order: The order of the polynomial to fit.
    :type order: int
    :param plot: If True, a plot of the operation happening will be shown.
        If a string is given that plot will be saved to the given file name.
    :type plot: bool or str


    Prepare some example data.

    >>> tr = obspy.read()[0].filter("highpass", freq=2)
    >>> tr.data += 6000 + 4 * tr.times() ** 2
    >>> tr.data -= 0.1 * tr.times() ** 3 + 0.00001 * tr.times() ** 5
    >>> data = tr.data

    Remove the trend.

    >>> polynomial(data, order=3, plot=True)  # doctest: +SKIP

    .. plot::

        import obspy
        from obspy.signal.detrend import polynomial

        tr = obspy.read()[0].filter("highpass", freq=2)
        tr.data += 6000 + 4 * tr.times() ** 2 - 0.1 * tr.times() ** 3 - \
            0.00001 * tr.times() ** 5

        polynomial(tr.data, order=3, plot=True)
    """
    # Convert data if it's not a floating point type.
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)

    # Check if the data is 1D or 2D and reshape if necessary
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Reshape to 2D with one channel
    elif data.ndim == 2 and data.shape[1] >= data.shape[0]:
        raise ValueError(
            "For 2D data, the number of channels (second axis) should be less than the number of data points (first axis).")
    elif data.ndim > 2:
        raise ValueError(
            "Data with more than two dimensions is not supported.")

    x = np.arange(data.shape[0])
    new_data = data.copy()
    fits = []

    for i in range(data.shape[1]):  # Iterate over rows (channels)
        fit = np.polyval(np.polyfit(x, data[:, i], deg=order), x)
        fits.append(fit)
        new_data[:, i] -= fit

    if plot:
        _plotting_helper(data, fits, plot)

    return new_data


def spline(data, order, dspline, plot=False):
    """
    Remove a trend by fitting splines.

    :param data: The data to detrend. Will be modified in-place.
    :type data: :class:`numpy.ndarray`
    :param order: The order/degree of the smoothing spline to fit.
        Must be 1 <= order <= 5.
    :type order: int
    :param dspline: The distance in samples between two spline nodes.
    :type dspline: int
    :param plot: If True, a plot of the operation happening will be shown.
        If a string is given that plot will be saved to the given file name.
    :type plot: bool or str


    .. note::

        In a real world application please make sure to use the convenience
        :meth:`obspy.core.trace.Trace.detrend` method.


    .. rubric:: Example

    >>> import obspy
    >>> from obspy.signal.detrend import spline

    Prepare some example data.

    >>> tr = obspy.read()[0].filter("highpass", freq=2)
    >>> tr.data += 6000 + 4 * tr.times() ** 2
    >>> tr.data -= 0.1 * tr.times() ** 3 + 0.00001 * tr.times() ** 5
    >>> data = tr.data

    Remove the trend.

    >>> spline(data, order=2, dspline=1000, plot=True)  # doctest: +SKIP

    .. plot::

        import obspy
        from obspy.signal.detrend import spline

        tr = obspy.read()[0].filter("highpass", freq=2)
        tr.data += 6000 + 4 * tr.times() ** 2 - 0.1 * tr.times() ** 3 - \
            0.00001 * tr.times() ** 5

        spline(tr.data, order=2, dspline=1000, plot=True)
    """
    # Convert data if it's not a floating point type.
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)

    # Check if the data is 1D or 2D and reshape if necessary
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Reshape to 2D with one channel
    elif data.ndim == 2 and data.shape[1] >= data.shape[0]:
        raise ValueError(
            "For 2D data, the number of channels (second axis) should be less than the number of data points (first axis).")
    elif data.ndim > 2:
        raise ValueError(
            "Data with more than two dimensions is not supported.")

    x = np.arange(data.shape[0])
    new_data = data.copy()
    fits = []

    for i in range(data.shape[1]):  # Iterate over columns (channels)
        splknots = np.arange(dspline / 2.0, len(x) -
                             dspline / 2.0 + 2, dspline)
        spl = LSQUnivariateSpline(x=x, y=data[:, i], t=splknots, k=order)
        fit = spl(x)
        fits.append(fit)
        print(fit.shape)
        new_data[:, i] -= fit

    if plot:
        _plotting_helper(data, fits, plot)

    return new_data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
