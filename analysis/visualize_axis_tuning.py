"""
Three-panel axis-tuning figure for a single unit: response vs. projection onto
its preferred axis (top), response vs. projection onto the orthogonal control
axis (left), and the joint 2D scatter of both projections colored by response
(main) -- each with a linear regression fit overlaid to show that response
tracks the preferred axis but is flat along the orthogonal one.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from analysis.sta_analysis import compute_sta, compute_orthogonal_variance


def plot_unit_axis_tuning(
    resp,
    params,
    method='sta',
    axis=None,
    orth_metrics=None,
    title=None,
    response_label='Response (a.u.)',
):
    """
    Args:
        resp: Response vector (n_stimuli,)
        params: Stimulus feature matrix (n_stimuli, n_features), e.g. PCA'd embeddings
        method: passed to compute_sta to derive the preferred axis, if axis is None
        axis: Precomputed preferred axis (n_features,); computed from resp/params if None
        orth_metrics: Precomputed dict from compute_orthogonal_variance; computed if None
        title: Figure suptitle; defaults to the selectivity index
        response_label: Axis label for the response (e.g. 'Firing rate (a.u.)')

    Returns:
        The created matplotlib Figure
    """
    if axis is None:
        axis, _ = compute_sta(resp, params, method=method, normalize=True)
    if orth_metrics is None:
        orth_metrics = compute_orthogonal_variance(resp, params, axis)

    proj_pref = orth_metrics['proj_pref']
    proj_orth = orth_metrics['proj_orth']
    ev_pref = orth_metrics['preferred_ev']
    ev_orth = orth_metrics['orthogonal_ev']

    coeffs_pref = np.polyfit(proj_pref, resp, 1)
    coeffs_orth = np.polyfit(proj_orth, resp, 1)

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(3, 3, figure=fig, hspace=0.05, wspace=0.05,
                  height_ratios=[1, 3, 0.1], width_ratios=[1, 3, 0.1])

    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)

    # Top panel: response vs. preferred-axis projection
    ax_top.scatter(proj_pref, resp, alpha=0.6, s=15, c='gray', edgecolors='none')
    x_sorted = np.sort(proj_pref)
    ax_top.plot(x_sorted, np.polyval(coeffs_pref, x_sorted), 'r-', linewidth=2)
    ax_top.set_ylabel(response_label, fontsize=10, fontweight='bold')
    ax_top.tick_params(labelbottom=False)
    ax_top.text(0.05, 0.95, f'EV = {ev_pref:.3f}', transform=ax_top.transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_top.grid(True, alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # Left panel: response vs. orthogonal-axis projection (axes swapped for vertical layout)
    ax_left.scatter(resp, proj_orth, alpha=0.6, s=15, c='gray', edgecolors='none')
    y_sorted = np.sort(proj_orth)
    ax_left.plot(np.polyval(coeffs_orth, y_sorted), y_sorted, 'r-', linewidth=2)
    ax_left.set_xlabel(response_label, fontsize=10, fontweight='bold')
    ax_left.set_ylabel('Distance along\northogonal axis (a.u.)', fontsize=9,
                        fontweight='bold', color='green')
    ax_left.tick_params(labelleft=False)
    ax_left.tick_params(axis='y', colors='green')
    ax_left.text(0.05, 0.95, f'EV = {ev_orth:.3f}', transform=ax_left.transAxes,
                 va='top', ha='left', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_left.grid(True, alpha=0.3)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.invert_xaxis()

    # Main panel: joint 2D scatter of both projections, colored by response
    scatter = ax_main.scatter(proj_pref, proj_orth, c=resp, cmap='viridis',
                               s=20, alpha=0.7, edgecolors='none')
    cbar_ax = fig.add_subplot(gs[1, 2])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(response_label, fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

    ax_main.set_xlabel('Distance along\npreferred axis (a.u.)', fontsize=10,
                        fontweight='bold', color='orange')
    ax_main.set_ylabel('Distance along\northogonal axis (a.u.)', fontsize=9,
                        fontweight='bold', color='green')
    ax_main.tick_params(axis='x', colors='orange')
    ax_main.tick_params(axis='y', colors='green')
    ax_main.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_main.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_main.grid(True, alpha=0.3)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    if title is None:
        title = f"Selectivity index = {orth_metrics['selectivity_index']:.3f}"
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

    return fig
