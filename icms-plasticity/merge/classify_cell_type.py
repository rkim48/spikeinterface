import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from spikeinterface import full as si


def get_trough_to_peak_ms(template):
    # assume template is one dimensional
    min_values_within_range = np.min(template[25:35], axis=0)
    global_min_index = np.argmin(min_values_within_range)
    primary_wvf = template[:, global_min_index]
    trough_to_peak_ms = np.argmax(primary_wvf[30:60]) / 30
    return trough_to_peak_ms


def classify_unit(cell_metrics):
    trough_to_peak_threshold = 0.425
    acg_tau_rise_threshold = 6
    rsquared_threshold = 0.22

    unit_ids = cell_metrics["unit_id"]
    trough_to_peak = cell_metrics["trough_to_peak"]
    acg_tau_rise = cell_metrics["acg_tau_rise"]
    acg_tau_decay = cell_metrics["acg_tau_decay"]
    rsquared_fit = cell_metrics["r_squared"]
    spike_counts = cell_metrics["spike_count"]

    # Initialize putative cell types as Pyramidal cells
    putative_cell_type = ["Pyramidal Cell"] * len(trough_to_peak)

    # Narrow Interneurons
    narrow_interneuron_indices = trough_to_peak <= trough_to_peak_threshold
    for idx in np.where(narrow_interneuron_indices)[0]:
        putative_cell_type[idx] = "Narrow Interneuron"

    # Wide Interneurons (only consider ACG tau rise if fit is good)
    wide_interneuron_indices = (
        (acg_tau_rise > acg_tau_rise_threshold)
        & (trough_to_peak > trough_to_peak_threshold)
        & (rsquared_fit > rsquared_threshold)
    )
    for idx in np.where(wide_interneuron_indices)[0]:
        putative_cell_type[idx] = "Wide Interneuron"

    # Convert to DataFrame for better visualization (optional)
    df = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "cell_type": putative_cell_type,
            "trough_to_peak": trough_to_peak,
            "acg_tau_rise": acg_tau_rise,
            "acg_tau_decay": acg_tau_decay,
            "r_squared": rsquared_fit,
            "spike_count": spike_counts,
        }
    )

    df.set_index("unit_id", inplace=True)

    return df


def fit_acg(acg_list):
    """
    Fits a triple exponential to the autocorrelogram with 0.5ms bins from -50ms to 50ms.
    """
    acg_array = np.array(acg_list)

    if acg_array.shape[1] < 101:
        raise ValueError("ACG arrays must have at least 101 data points.")

    acg_array[:, 99:102] = 0
    offset = 100
    x = np.arange(1, 101) / 2

    # Variables
    a0 = [100, 2, 20, 2, 0.5, 1, 2, 2]
    lb = [10, 0.1, 0, 0, -30, 0, 0.1, 0]
    ub = [2000, 50, 2000, 15, 50, 20, 10, 100]

    fit_params_array = np.full((8, acg_array.shape[0]), np.nan)
    rsquare = np.full(acg_array.shape[0], np.nan)

    def fit_equation(x, a, b, c, d, e, f, g, h):
        return np.maximum(
            c * (np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b)) +
            h * np.exp(-(x - f) / g) + e,
            0,
        )

    spike_count_values = []

    for i in range(acg_array.shape[0]):
        try:
            valid_indices = np.arange(offset, offset + len(x)).astype(int)
            y_data = acg_array[i, valid_indices]
            spike_count_values.append(sum(y_data))

            popt, _ = curve_fit(fit_equation, x, y_data,
                                p0=a0, bounds=(lb, ub), maxfev=5000)

            fit_params_array[:, i] = popt
            residuals = y_data - fit_equation(x, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            rsquare[i] = 1 - (ss_res / ss_tot)

        except RuntimeError as e:
            print(f"Error fitting unit {i}: {e}")
            # Handle fitting error by appending NaN
        except Exception as e:
            print(f"Unexpected error for unit {i}: {e}")
            # Handle unexpected error by appending NaN

    fit_params = {
        "acg_tau_decay": fit_params_array[0, :],
        "acg_tau_rise": fit_params_array[1, :],
        "acg_c": fit_params_array[2, :],
        "acg_d": fit_params_array[3, :],
        "acg_asymptote": fit_params_array[4, :],
        "acg_refrac": fit_params_array[5, :],
        "acg_tau_burst": fit_params_array[6, :],
        "acg_h": fit_params_array[7, :],
        "acg_fit_rsquare": rsquare,
    }

    return fit_params, spike_count_values


def classify_units_into_cell_types(analyzer):
    """
    Parameters
    ----------
    analyzer : sorting analyzer with accepted unit ids only

    Returns
    -------
    result : dict containing unit ids, putative cell types, templates, acgs, acg fits, and other information
    """
    # Classify child units into putative cell types
    templates_dict = {}
    for unit_id in analyzer.unit_ids:
        templates = analyzer.get_extension("templates")
        template = templates.get_unit_template(unit_id)
        templates_dict[unit_id] = template

    # if not analyzer.has_extension("correlograms"):
    analyzer.compute("correlograms", window_ms=100, bin_ms=0.5)

    ccgs_ext = analyzer.get_extension("correlograms")
    ccgs, time_bins = ccgs_ext.get_data()

    acg_dict = {unit_id: ccgs[i, i, :]
                for i, unit_id in enumerate(analyzer.unit_ids)}

    fit_params, spike_count_values = fit_acg(list(acg_dict.values()))
    trough_to_peak_values = {unit_id: get_trough_to_peak_ms(
        template) for unit_id, template in templates_dict.items()}
    spike_count_values = {
        unit_id: count for unit_id, count in zip(analyzer.unit_ids, analyzer.sorting.get_total_num_spikes().values())
    }

    cell_metrics = {
        "unit_id": np.array(analyzer.unit_ids),
        "trough_to_peak": np.array([trough_to_peak_values[unit_id] for unit_id in analyzer.unit_ids]),
        "acg_tau_rise": np.array(fit_params["acg_tau_rise"]),
        "acg_tau_decay": np.array(fit_params["acg_tau_decay"]),
        "r_squared": np.array(fit_params["acg_fit_rsquare"]),
        "spike_count": np.array([spike_count_values[unit_id] for unit_id in analyzer.unit_ids]),
    }

    df = classify_unit(cell_metrics)

    fit_params_dict = {
        unit_id: {key: fit_params[key][i] for key in fit_params} for i, unit_id in enumerate(analyzer.unit_ids)
    }

    result = {
        "df": df,
        "acgs": acg_dict,
        "fit_params": fit_params_dict,
    }

    return result


# %% Plotting functions


def plot_acg_with_fit(result, unit_id, ax=None):
    acg = result["acgs"][unit_id]
    fit_params = result["fit_params"][unit_id]
    df = result["df"]

    if "unit_id" in df.index.names:
        df = df.reset_index()

    cell_type = df.loc[df["unit_id"] == unit_id, "cell_type"].values[0]
    cell_metrics = df.set_index("unit_id").loc[unit_id]

    a = fit_params["acg_tau_decay"]
    b = fit_params["acg_tau_rise"]
    c = fit_params["acg_c"]
    d = fit_params["acg_d"]
    e = fit_params["acg_asymptote"]
    f = fit_params["acg_refrac"]
    g = fit_params["acg_tau_burst"]
    h = fit_params["acg_h"]
    r_sq = fit_params["acg_fit_rsquare"]
    rsquared_threshold = 0.22

    x = np.arange(1, 101) / 2
    fit = np.maximum(
        c * (np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b)) +
        h * np.exp(-(x - f) / g) + e,
        0,
    )

    use_own_ax = False
    if ax is None:
        use_own_ax = True
        fig, ax = plt.figure()

    # Plot the ACG and fit
    acg_x = np.arange(100) / 2  # assuming 0.5 ms bins for ACG
    ax.plot(acg_x, acg[100:], label="ACG")
    if r_sq > rsquared_threshold:
        ax.plot(acg_x, fit, label="Fit", color="k")
        ax.legend()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Count")

    if use_own_ax:
        ax.set_title(
            f'Unit ID: {unit_id}, Cell Type: {cell_type}, Trough to Peak: {
                cell_metrics["trough_to_peak"]:.2f} ms\n'
            f"ACG Tau Rise: {b:.2f} ms, $R^2$: {r_sq:.2f}"
        )
    else:
        ax.set_title(
            f'{cell_type}, T2P: {cell_metrics["trough_to_peak"]:.2f} ms\n'
            f"ACG Tau Rise: {b:.2f} ms, $R^2$: {r_sq:.2f}"
        )

    plt.show()


def plot_primary_wvf_with_acg_with_fit(result, unit_id):
    template = result["templates"][unit_id]
    x = np.arange(1, 101) / 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    template_x = np.arange(template.shape[0]) / 30
    min_values_within_range = np.min(template[20:40, :], axis=0)
    global_min_index = np.argmin(min_values_within_range)
    axes[0].plot(template_x, template[:, global_min_index],
                 label="Template (Primary Channel)")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")

    plot_acg_with_fit(result, unit_id, ax=axes[1])

    plt.show()
