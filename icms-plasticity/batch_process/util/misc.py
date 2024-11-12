from spikeinterface import full as si
import numpy as np


# def get_sparse_template(analyzer, unit_id):
#     analyzer.sparsity = si.compute_sparsity(analyzer, method="radius", radius_um=60)

#     analyzer.compute("random_spikes")
#     analyzer.compute("waveforms")
#     analyzer.compute("templates")

#     templates_ext = analyzer.get_extension("templates")
#     template = templates_ext.get_unit_template(unit_id)


def get_unit_id_properties(sorting):
    property_keys = sorting.get_property_keys()
    unit_ids = sorting.unit_ids
    unit_id_properties = {}
    for unit_id in unit_ids:
        unit_id_properties[unit_id] = []
        for key in property_keys:
            if sorting.get_unit_property(unit_id, key):
                unit_id_properties[unit_id].append(key)
    return unit_id_properties


def get_bad_events_dict(bad_analyzer, bad_ids, sorting):
    # get dict that contains channel sample pair where bad events occur
    bad_extremum_ch_inds = si.get_template_extremum_channel(bad_analyzer, mode="at_index", outputs="index")
    bad_sorting = sorting.select_units(unit_ids=bad_ids, renamed_unit_ids=bad_ids)
    bad_sv = bad_sorting.to_spike_vector(extremum_channel_inds=bad_extremum_ch_inds)

    channel_indices = bad_sv["channel_index"]
    sample_indices = bad_sv["sample_index"]
    bad_events_dict = {}
    for channel_index, sample in zip(channel_indices, sample_indices):
        if channel_index in bad_events_dict:
            bad_events_dict[channel_index].append(sample)
        else:
            bad_events_dict[channel_index] = [sample]

    return bad_events_dict


def align_waveforms(we, sorting):
    unit_peaks_shift = {}
    for unit_id in we.unit_ids:
        template = we.get_template(unit_id)
        min_values_within_range = np.min(template[20:40, :], axis=0)
        global_min_index = np.argmin(min_values_within_range)
        primary_wvf = template[:, global_min_index]
        peak_idx = np.argmin(primary_wvf)
        shift = 30 - peak_idx
        unit_peaks_shift[unit_id] = shift

        # plt.figure()
        # template = we.get_template(unit_id)
        # plt.plot(template)
        # plt.plot(primary_wvf, linewidth=3)
        # plt.axvline(30)
        # plt.title(unit_id)

    align_sorting = si.align_sorting(sorting, unit_peaks_shift)
    return align_sorting


def get_adjusted_depth(depth, animal_id):
    match animal_id:
        case "ICMS92":
            ch_out = 1
        case "ICMS93":
            ch_out = 1
        case "ICMS98":
            ch_out = 1
        case "ICMS100":
            ch_out = 1
        case "ICMS101":
            ch_out = 1
        case _:
            print(f"Animal ID '{animal_id}' not found.")
            return None

    pitch_um = 60
    angle_from_vertical_deg = 30
    angle_from_vertical_rad = np.radians(angle_from_vertical_deg)
    length_out = pitch_um * ch_out
    adjusted_depth = (depth - length_out) * np.cos(angle_from_vertical_rad)

    return adjusted_depth
