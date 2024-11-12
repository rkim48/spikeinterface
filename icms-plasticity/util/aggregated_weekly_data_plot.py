import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from batch_process.postprocessing.responses import response_plotting_util as rpu
from scipy.stats import mannwhitneyu
import pandas as pd


def plot_aggregated_weekly_data_with_bootstrapped_ci(df, var1="z_score", var2="pulse_mean_fr", animal_id="Aggregated", ax1=None, ax2=None):
    color_map = rpu.get_stim_colormap()

    if ax1 is None or ax2 is None:
        raise ValueError("Both ax1 and ax2 must be provided.")

    # Create a new column for 'weeks_relative' by dividing 'days_relative' by 7 and flooring it
    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
    len_x = len(df['weeks_relative'].unique())

    # Initialize p-values dictionaries
    p_values_var1 = {}
    p_values_var2 = {}

    # Get unique stimulation currents
    stim_currents = sorted(df['stim_current'].unique())
    # Offset positions for boxplots
    current_offsets = np.linspace(-0.3, 0.3, len(stim_currents))

    # Boxplot for var1 without median lines
    for idx, stim_current in enumerate(stim_currents):
        current_data_var1 = df[df['stim_current'] == stim_current]

        week_0_var1 = current_data_var1[current_data_var1['weeks_relative'] == 0][var1]
        week_last_var1 = current_data_var1[current_data_var1['weeks_relative'] == 5][var1]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var1, week_last_var1, alternative='less')
        p_values_var1[stim_current] = p_val

        # Calculate and plot boxplot
        aggregated_data_var1 = current_data_var1.groupby('weeks_relative').apply(
            lambda x: pd.Series({
                'median': x[var1].median(),
                'ci_low': bootstrap_ci(x[var1])[0],
                'ci_high': bootstrap_ci(x[var1])[1]
            })
        ).reset_index()

        # Shift the positions slightly for boxplots
        box_positions = aggregated_data_var1['weeks_relative'] + \
            current_offsets[idx]

        # Box plot for var1
        sns.boxplot(x='weeks_relative', y=var1, data=current_data_var1, ax=ax1,
                    color=color_map[stim_current], width=0.2,
                    boxprops=dict(alpha=0.8), showfliers=False,
                    positions=box_positions)

        # Add star if p <= 0.05 above the last boxplot (week 5)
        if np.round(p_val, 2) <= 0.05:
            last_median = aggregated_data_var1[aggregated_data_var1['weeks_relative']
                                               == 5]['median'].values[0]
            ax1.text(5 + current_offsets[idx], last_median + 0.5, '*',
                     color='k', fontsize=15, ha='center')

    # Replace underscores with spaces for better formatting
    var1_label = var1.replace('_', ' ').capitalize()

    ax1.set_title(f"Median Z-Score with 95% CI Across Weeks", fontsize=10)
    ax1.set_ylabel(f"Z-Score")
    ax1.set_xlabel("Weeks Relative to First Session")

    ax1.set_xticks(np.arange(len_x))  # Set ticks at the weeks
    ax1.set_xticklabels(np.arange(len_x))

    # Boxplot for var2 without median lines
    for idx, stim_current in enumerate(stim_currents):
        current_data_var2 = df[df['stim_current'] == stim_current]

        week_0_var2 = current_data_var2[current_data_var2['weeks_relative'] == 0][var2]
        week_last_var2 = current_data_var2[current_data_var2['weeks_relative'] == 5][var2]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var2, week_last_var2, alternative='less')
        p_values_var2[stim_current] = p_val

        # Calculate and plot boxplot
        aggregated_data_var2 = current_data_var2.groupby('weeks_relative').apply(
            lambda x: pd.Series({
                'median': x[var2].median(),
                'ci_low': bootstrap_ci(x[var2])[0],
                'ci_high': bootstrap_ci(x[var2])[1]
            })
        ).reset_index()

        # Shift the positions slightly for boxplots
        box_positions = aggregated_data_var2['weeks_relative'] + \
            current_offsets[idx]

        # Box plot for var2
        sns.boxplot(x='weeks_relative', y=var2, data=current_data_var2, ax=ax2,
                    color=color_map[stim_current], width=0.2,
                    boxprops=dict(alpha=0.8), showfliers=False,
                    positions=box_positions)

        # Add star if p <= 0.05 above the last boxplot (week 5)
        if np.round(p_val, 2) <= 0.05:
            last_median = aggregated_data_var2[aggregated_data_var2['weeks_relative']
                                               == 5]['median'].values[0]
            ax2.text(5 + current_offsets[idx], last_median + 0.5, '*',
                     color='k', fontsize=15, ha='center')

    # Replace underscores with spaces for better formatting
    var2_label = var2.replace('_', ' ').capitalize()
    ax1.set_xlim([-0.4, 5.4])
    ax1.set_ylim([0, 30])

    ax2.set_xlim([-0.4, 5.4])

    ax2.set_title(
        f"Median Pulse Window Firing Rate with 95% CI Across Weeks", fontsize=10)
    ax2.set_ylabel(f"Firing Rate (Hz)")
    ax2.set_xlabel("Weeks Relative to First Session")

    ax2.set_xticks(np.arange(len_x))  # Set ticks at the weeks
    ax2.set_xticklabels(np.arange(len_x))

    # Create custom legend using Line2D
    custom_lines = [Line2D([0], [0], color=color_map[stim_current], lw=4)
                    for stim_current in stim_currents]
    labels = [f"{stim_current} µA" for stim_current in stim_currents]

    # Add legend to both axes
    # ax1.legend(custom_lines, labels, loc="best")
    ax2.legend(custom_lines, labels, loc="best")

    plt.tight_layout()
    return p_values_var1, p_values_var2


def plot_aggregated_weekly_data_with_bootstrapped_ci(df, var1="z_score", var2="pulse_mean_fr", animal_id="Aggregated", ax1=None, ax2=None):
    color_map = rpu.get_stim_colormap()

    if ax1 is None or ax2 is None:
        raise ValueError("Both ax1 and ax2 must be provided.")

    # Create a new column for 'weeks_relative' by dividing 'days_relative' by 7 and flooring it
    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
    len_x = len(df['weeks_relative'].unique())

    # Initialize p-values dictionaries
    p_values_var1 = {}
    p_values_var2 = {}

    # Get unique stimulation currents
    stim_currents = sorted(df['stim_current'].unique())
    # Offset positions for error bars
    current_offsets = np.linspace(-0.2, 0.2, len(stim_currents))

    # Line plot for var1 (z_score) with 95% CI as error bars
    for idx, stim_current in enumerate(stim_currents):
        current_data_var1 = df[df['stim_current'] == stim_current]

        week_0_var1 = current_data_var1[current_data_var1['weeks_relative'] == 0][var1]
        week_last_var1 = current_data_var1[current_data_var1['weeks_relative'] == 5][var1]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var1, week_last_var1, alternative='less')
        p_values_var1[stim_current] = p_val

        # Calculate median and CI for each week
        aggregated_data_var1 = current_data_var1.groupby('weeks_relative').apply(
            lambda x: pd.Series({
                'median': x[var1].median(),
                'ci_low': bootstrap_ci(x[var1])[0],
                'ci_high': bootstrap_ci(x[var1])[1]
            })
        ).reset_index()

        # Plot median with error bars (95% CI)
        ax1.errorbar(
            # Horizontal offset for each current
            aggregated_data_var1['weeks_relative'] + current_offsets[idx],
            aggregated_data_var1['median'],
            yerr=[aggregated_data_var1['median'] - aggregated_data_var1['ci_low'],
                  aggregated_data_var1['ci_high'] - aggregated_data_var1['median']],
            fmt='-o',
            label=f"{stim_current} µA",
            color=color_map[stim_current],
            capsize=4
        )

        # Add star if p <= 0.05 above the last point (week 5)
        if np.round(p_val, 2) <= 0.05:
            last_median = aggregated_data_var1[aggregated_data_var1['weeks_relative']
                                               == 5]['median'].values[0]
            ax1.text(5 + current_offsets[idx], last_median + 0.5, '*',
                     color='k', fontsize=15, ha='center')

    # Replace underscores with spaces for better formatting
    var1_label = var1.replace('_', ' ').capitalize()

    ax1.set_title(f"Median Z-Score with 95% CI Across Weeks", fontsize=10)
    ax1.set_ylabel(f"Z-Score")
    ax1.set_xlabel("Weeks Relative to First Session")

    ax1.set_xticks(np.arange(len_x))  # Set ticks at the weeks
    ax1.set_xticklabels(np.arange(len_x))

    # Line plot for var2 (pulse_mean_fr) with 95% CI as error bars
    for idx, stim_current in enumerate(stim_currents):
        current_data_var2 = df[df['stim_current'] == stim_current]

        week_0_var2 = current_data_var2[current_data_var2['weeks_relative'] == 0][var2]
        week_last_var2 = current_data_var2[current_data_var2['weeks_relative'] == 5][var2]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var2, week_last_var2, alternative='less')
        p_values_var2[stim_current] = p_val

        # Calculate median and CI for each week
        aggregated_data_var2 = current_data_var2.groupby('weeks_relative').apply(
            lambda x: pd.Series({
                'median': x[var2].median(),
                'ci_low': bootstrap_ci(x[var2])[0],
                'ci_high': bootstrap_ci(x[var2])[1]
            })
        ).reset_index()

        # Plot median with error bars (95% CI)
        ax2.errorbar(
            # Horizontal offset for each current
            aggregated_data_var2['weeks_relative'] + current_offsets[idx],
            aggregated_data_var2['median'],
            yerr=[aggregated_data_var2['median'] - aggregated_data_var2['ci_low'],
                  aggregated_data_var2['ci_high'] - aggregated_data_var2['median']],
            fmt='-o',
            label=f"{stim_current} µA",
            color=color_map[stim_current],
            capsize=4
        )

        # Add star if p <= 0.05 above the last point (week 5)
        if np.round(p_val, 2) <= 0.05:
            last_median = aggregated_data_var2[aggregated_data_var2['weeks_relative']
                                               == 5]['median'].values[0]
            ax2.text(5 + current_offsets[idx], last_median + 0.5, '*',
                     color='k', fontsize=15, ha='center')

    # Replace underscores with spaces for better formatting
    var2_label = var2.replace('_', ' ').capitalize()
    ax1.set_xlim([-0.4, 5.4])
    ax1.set_ylim([0, 15])

    ax2.set_xlim([-0.4, 5.4])
    ax2.set_ylim([0, 50])

    ax2.set_title(
        f"Median Pulse Window Firing Rate with 95% CI Across Weeks", fontsize=10)
    ax2.set_ylabel(f"Firing Rate (Hz)")
    ax2.set_xlabel("Weeks Relative to First Session")

    ax2.set_xticks(np.arange(len_x))  # Set ticks at the weeks
    ax2.set_xticklabels(np.arange(len_x))

    # Create custom legend using Line2D
    custom_lines = [Line2D([0], [0], color=color_map[stim_current], lw=4)
                    for stim_current in stim_currents]
    labels = [f"{stim_current} µA" for stim_current in stim_currents]

    # Add legend to both axes
    ax1.legend(custom_lines, labels, loc="best")
    ax2.legend(custom_lines, labels, loc="best")

    plt.tight_layout()
    return p_values_var1, p_values_var2


def plot_aggregated_weekly_data_with_iqr(df, var1="z_score", var2=None, last_week=5, ax1=None, ax2=None):
    color_map = rpu.get_stim_colormap()

    if ax1 is None:
        raise ValueError("ax1 must be provided.")

    # Create a new column for 'weeks_relative' by dividing 'days_relative' by 7 and flooring it
    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
    df = df[df['weeks_relative'] <= last_week]
    len_x = last_week + 1

    # Initialize p-values dictionaries
    p_values_var1 = {}
    p_values_var2 = {}

    # Get unique stimulation currents
    stim_currents = sorted(df['stim_current'].unique())
    # Offset positions for error bars
    current_offsets = np.linspace(-0.2, 0.2, len(stim_currents))

    # Line plot for var1 (z_score) with IQR as error bars
    for idx, stim_current in enumerate(stim_currents):
        current_data_var1 = df[df['stim_current'] == stim_current]

        week_0_var1 = current_data_var1[current_data_var1['weeks_relative'] == 0][var1]
        week_last_var1 = current_data_var1[current_data_var1['weeks_relative']
                                           == last_week][var1]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var1, week_last_var1, alternative='less')
        p_values_var1[stim_current] = p_val

        # Calculate median and IQR for each week
        aggregated_data_var1 = current_data_var1.groupby('weeks_relative').apply(
            lambda x: pd.Series({
                'median': x[var1].median(),
                'iqr_low': np.percentile(x[var1], 25),
                'iqr_high': np.percentile(x[var1], 75)
            })
        ).reset_index()

        # Plot median with error bars (IQR)
        ax1.errorbar(
            aggregated_data_var1['weeks_relative'] + current_offsets[idx],
            aggregated_data_var1['median'],
            yerr=[aggregated_data_var1['median'] - aggregated_data_var1['iqr_low'],
                  aggregated_data_var1['iqr_high'] - aggregated_data_var1['median']],
            fmt='-o',
            label=f"{stim_current} µA",
            color=color_map[stim_current],
            capsize=4
        )

        # Add star if p <= 0.05 above the last point
        if np.round(p_val, 2) <= 0.05:
            last_median = aggregated_data_var1[aggregated_data_var1['weeks_relative']
                                               == last_week]['median'].values[0]
            ax1.text(last_week + current_offsets[idx], last_median +
                     0.5, '*', color='k', fontsize=15, ha='center')

    # Replace underscores with spaces for better formatting
    ax1.set_title("ICMS Train Modulation Score Across Weeks", fontsize=8)
    ax1.set_ylabel(f"Z-score", fontsize=8)
    ax1.set_xlabel("Weeks Relative to First Session", fontsize=8)

    ax1.set_xticks(np.arange(len_x))  # Set ticks at the weeks
    ax1.set_xticklabels(np.arange(len_x), fontsize=8)
    ax1.tick_params(axis='y', labelsize=8)  # Set y-axis tick font size

    # Create custom legend using Line2D
    custom_lines = [Line2D([0], [0], color=color_map[stim_current], lw=4)
                    for stim_current in stim_currents]
    labels = [f"{stim_current} µA" for stim_current in stim_currents]

    # Add legend to ax1
    ax1.legend(custom_lines, labels, loc="upper left", fontsize=6)

    # Only plot var2 if provided
    if var2 is not None:
        if ax2 is None:
            raise ValueError("ax2 must be provided when var2 is specified.")

        # Line plot for var2 with IQR as error bars
        for idx, stim_current in enumerate(stim_currents):
            current_data_var2 = df[df['stim_current'] == stim_current]

            week_0_var2 = current_data_var2[current_data_var2['weeks_relative'] == 0][var2]
            week_last_var2 = current_data_var2[current_data_var2['weeks_relative']
                                               == last_week][var2]

            # Perform Mann-Whitney U test
            stat, p_val = mannwhitneyu(
                week_0_var2, week_last_var2, alternative='less')
            p_values_var2[stim_current] = p_val

            # Calculate median and IQR for each week
            aggregated_data_var2 = current_data_var2.groupby('weeks_relative').apply(
                lambda x: pd.Series({
                    'median': x[var2].median(),
                    'iqr_low': np.percentile(x[var2], 25),
                    'iqr_high': np.percentile(x[var2], 75)
                })
            ).reset_index()

            # Plot median with error bars (IQR)
            ax2.errorbar(
                aggregated_data_var2['weeks_relative'] + current_offsets[idx],
                aggregated_data_var2['median'],
                yerr=[aggregated_data_var2['median'] - aggregated_data_var2['iqr_low'],
                      aggregated_data_var2['iqr_high'] - aggregated_data_var2['median']],
                fmt='-o',
                label=f"{stim_current} µA",
                color=color_map[stim_current],
                capsize=4
            )

            # Add star if p <= 0.05 above the last point
            if np.round(p_val, 2) <= 0.05:
                last_median = aggregated_data_var2[aggregated_data_var2['weeks_relative']
                                                   == last_week]['median'].values[0]
                ax2.text(
                    last_week + current_offsets[idx], last_median + 0.5, '*', color='k', fontsize=8, ha='center')

        ax2.set_xlim([-0.4, last_week + .4])
        ax2.set_ylim([0, 50])

        ax2.set_title(f"Median {var2.replace(
            '_', ' ').capitalize()} with IQR Across Weeks", fontsize=10)
        ax2.set_ylabel(f"{var2.replace('_', ' ').capitalize()}")
        ax2.set_xlabel("Weeks Relative to First Session")

        ax2.set_xticks(np.arange(len_x))  # Set ticks at the weeks
        ax2.set_xticklabels(np.arange(len_x))

        # Add legend to ax2
        ax2.legend(custom_lines, labels, loc="best")

    plt.tight_layout()
    return p_values_var1, p_values_var2
