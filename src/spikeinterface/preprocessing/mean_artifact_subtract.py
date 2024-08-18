import numpy as np
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from ..core.core_tools import define_function_from_class


class MeanArtifactSubtractedRecording(BasePreprocessor):
    """
    Subtract the mean or median of post-stimulus pulse windows in a trial.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be processed.
    list_triggers : list of list
        List of lists containing pulse times in samples for each trial.
    post_stim_window_ms : float
        Post-stimulus window in milliseconds for averaging.
    mode : str, optional
        'mean' or 'median', by default 'mean'.
    """

    name = "mean_artifact_subtract"

    def __init__(self, recording, list_triggers, post_stim_window_ms, mode="mean"):
        self._list_triggers = list_triggers
        self._post_stim_window_ms = post_stim_window_ms
        self._mode = mode
        self._sf = recording.get_sampling_frequency()

        BasePreprocessor.__init__(self, recording)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(
                MeanArtifactSubtractedRecordingSegment(parent_segment, list_triggers, self._post_stim_window_ms, mode)
            )

        self._kwargs = {
            "recording": recording,
            "list_triggers": list_triggers,
            "post_stim_window_ms": post_stim_window_ms,
            "mode": mode,
        }


class MeanArtifactSubtractedRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, list_triggers, post_stim_window_ms, mode):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self._list_triggers = list_triggers
        self._post_stim_window_samples = post_stim_window_ms * 30
        self._mode = mode

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)

        for trial_triggers in self._list_triggers:
            # Calculate mean or median for the post-stimulus windows
            post_stim_traces = []
            for ts in trial_triggers:
                end_ts = ts + self._post_stim_window_samples
                window = traces[ts:end_ts, :]
                post_stim_traces.append(window)

            post_stim_traces = np.array(post_stim_traces)
            if self._mode == "mean":
                avg_trace = np.mean(post_stim_traces, axis=0)
            else:  # median
                avg_trace = np.median(post_stim_traces, axis=0)

            # Subtract the average from each pulse window
            for ts in trial_triggers:
                end_ts = ts + self._post_stim_window_samples
                traces[ts:end_ts, :] -= avg_trace.astype(traces.dtype)

        return traces


# function for API
mean_artifact_subtract = define_function_from_class(
    source_class=MeanArtifactSubtractedRecording, name="mean_artifact_subtract"
)
