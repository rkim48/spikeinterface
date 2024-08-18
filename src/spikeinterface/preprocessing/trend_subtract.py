import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from numpy.polynomial import Polynomial
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from ..core.core_tools import define_function_from_class
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore", np.RankWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TrendSubtractedRecording(BasePreprocessor):
    """
    Subtract the polynomial or spline fit of post-stimulus pulse windows in a trial.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be processed.
    list_triggers : list of list
        List of lists containing pulse times in samples for each trial.
    post_pulse_start_ms : float
        The time in samples after pulse to start the fit.
    post_pulse_end_ms : float
        The time in samples after pulse to end the fit.
    mode : str, optional
        'poly' or 'spline', by default 'poly'.
    poly_order : int, optional
    spline_order: int, optional
    dspline: int, optional
        samples between spline points
    """

    name = "trend_subtract"
    installed = True

    def __init__(
        self,
        recording,
        list_triggers,
        post_pulse_start_ms,
        post_pulse_end_ms,
        mode="poly",
        poly_order=3,
        spline_order=3,
        dspline=1000,
    ):
        assert mode in ["poly", "spline"], "Mode must be either 'poly' or 'spline'"

        self._list_triggers = list_triggers
        self._sf = recording.get_sampling_frequency()
        self._post_pulse_start_ms = post_pulse_start_ms
        self._post_pulse_end_ms = post_pulse_end_ms
        self._mode = mode
        self._poly_order = poly_order
        self._spline_order = spline_order
        self._dspline = dspline

        BasePreprocessor.__init__(self, recording)
        for parent_segment in recording._recording_segments:
            recording_segment = TrendSubtractedRecordingSegment(
                parent_segment,
                list_triggers,
                post_pulse_start_ms,
                post_pulse_end_ms,
                mode,
                poly_order,
                spline_order,
                dspline,
            )
            self.add_recording_segment(recording_segment)

        self._kwargs = {
            "recording": recording,
            "list_triggers": list_triggers,
            "post_pulse_start_ms": post_pulse_start_ms,
            "post_pulse_end_ms": post_pulse_end_ms,
            "mode": mode,
            "poly_order": poly_order,
            "spline_order": spline_order,
            "dspline": dspline,
        }


class TrendSubtractedRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        list_triggers,
        post_pulse_start_ms,
        post_pulse_end_ms,
        mode,
        poly_order,
        spline_order,
        dspline,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self._list_triggers = [[np.int64(ts) for ts in trial] for trial in list_triggers]
        self._sf = parent_recording_segment.sampling_frequency
        self._post_pulse_start_samples = int(self._sf * (post_pulse_start_ms / 1000.0))
        self._post_pulse_end_samples = int(self._sf * (post_pulse_end_ms / 1000.0))
        self._mode = mode
        self._poly_order = poly_order
        self._spline_order = spline_order
        self._dspline = dspline

    @staticmethod
    def _plotting_helper(data, fits):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5))
        plt.subplots_adjust(hspace=0)

        # Plot original data and fits for each channel
        for i in range(data.shape[1]):
            axes[0].plot(data[:, i], label=f"Channel {i+1} Original", color="k")
            axes[0].plot(fits[i], label=f"Channel {i+1} Fit", lw=1, color="r")
        axes[0].label_outer()

        # Plot residuals for each channel
        for i in range(data.shape[1]):
            axes[1].plot(data[:, i] - fits[i])

        # axes[1].legend(loc="best")
        axes[1].label_outer()
        axes[1].set_xlabel("Samples")

        plt.tight_layout(h_pad=0)
        plt.show()

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)

        self.fits = []
        num_channels = np.shape(traces)[1]

        for trial_triggers in self._list_triggers:
            for ts in trial_triggers:
                if start_frame <= ts + self._post_pulse_start_samples < end_frame:
                    ts1 = max(ts + self._post_pulse_start_samples - start_frame, 0)
                    ts2 = min(ts + self._post_pulse_end_samples - start_frame, traces.shape[0])
                    for channel_idx in range(num_channels):

                        # Extract the post-stimulus window
                        window = traces[ts1:ts2, channel_idx]
                        x = np.arange(ts1, ts2)

                        # Perform trend subtraction
                        if self._mode == "poly":
                            try:
                                fit = np.polyval(np.polyfit(x, window, deg=4), x)
                            except np.RankWarning:
                                # Fallback to order 3
                                fit = np.polyval(np.polyfit(x, window, deg=self._poly_order), x)
                        elif self._mode == "spline":
                            splknots = np.arange(self._dspline / 2.0, len(x) - self._dspline / 2.0 + 2, self._dspline)
                            spl = LSQUnivariateSpline(x=x, y=window, t=splknots, k=self._spline_order)
                            fit = spl(x)
                            self.fits.append(fit)
                        elif self._mode == "exp":
                            a, b = np.polyfit(x, np.log(window), 1, w=np.sqrt(window))
                            fit = np.exp(b) * np.exp(a * x)
                        # Apply the trend subtraction to the post-stimulus window
                        traces[ts1:ts2, channel_idx] = window - fit.astype(window.dtype)
        return traces

    def plot_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        self._plotting_helper(traces, self.fits)


# function for API
trend_subtract = define_function_from_class(source_class=TrendSubtractedRecording, name="trend_subtract")
