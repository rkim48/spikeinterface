import numpy as np
from spikeinterface.postprocessing import (
    compute_spike_amplitudes, compute_unit_locations,
    compute_template_similarity, compute_correlograms
)
from spikeinterface import full as si
from spikeinterface import qualitymetrics as qm


def auto_curate(we, sorting, job_kwargs):

    # keep units whose max absolute amplitude < 1000 uV
    extrema = si.get_template_extremum_amplitude(we)
    unit_ids = we.unit_ids
    keep_unit_ids1 = []

    for i, extremum in enumerate(extrema):
        if extremum < 600:
            keep_unit_ids1.append(unit_ids[i])
    keep_unit_ids1 = np.array(keep_unit_ids1)
    # other metrics (allow more units through since stim violates certain assumptions)
    amplitude_cutoff_thresh = 0.2  # originally 0.1
    isi_violations_ratio_thresh = 2  # originally 1
    presence_ratio_thresh = 0.6  # originally 0.9
    _ = compute_spike_amplitudes(we, **job_kwargs)
    _ = compute_unit_locations(we)
    _ = compute_template_similarity(we)
    _ = compute_correlograms(we)

    our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & \
        (isi_violations_ratio < {isi_violations_ratio_thresh}) & \
            (presence_ratio > {presence_ratio_thresh})"
    metrics = qm.compute_quality_metrics(we,
                                         metric_names=['firing_rate',
                                                       'presence_ratio',
                                                       'snr', 'isi_violation',
                                                       'amplitude_cutoff'])
    # Apply query
    keep_units2 = metrics.query(our_query)
    keep_unit_ids2 = keep_units2.index.values
    # Get intersection of both unit id arrays
    keep_unit_ids = np.intersect1d(keep_unit_ids1, keep_unit_ids2)
    remove_unit_ids = np.setdiff1d(unit_ids, keep_unit_ids)

    we_curated = we.select_units(
        keep_unit_ids, new_folder=None)
    sorting_curated = sorting.select_units(keep_unit_ids)
    return we_curated, sorting_curated, remove_unit_ids
