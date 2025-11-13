import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
from spike_metrics import compute_cluster_metrics

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _density_image(W, ax, w=400, h=240, interpolation_factor=200,
                   interpolation_kind="linear", log=True, cmap="inferno",
                   samplerate_hz=None):
    """Fast joint-PDF density of waveforms W(n,T) directly on ax."""
    n, T = W.shape
    if n == 0:
        ax.set_axis_off(); return
    x_min, x_max = 0.0, float(T - 1)
    y_min, y_max = float(W.min()), float(W.max())
    pad = 1e-6 * (y_max - y_min if y_max > y_min else 1.0)
    y_min -= pad; y_max += pad

    x_edges = np.linspace(x_min, x_max, w + 1)
    y_edges = np.linspace(y_min, y_max, h + 1)

    Ti = int(max(2, min(T * interpolation_factor, w)))
    x_orig = np.arange(T, dtype=np.float64)
    x_hi   = np.linspace(x_min, x_max, Ti, endpoint=True)
    f  = interp1d(x_orig, W, axis=1, kind=interpolation_kind, assume_sorted=True)
    hi = f(x_hi)

    t_flat = np.tile(x_hi, n)
    a_flat = hi.ravel()
    H, _, _ = np.histogram2d(t_flat, a_flat, bins=(x_edges, y_edges))
    H = H.T

    bin_area = ((x_max - x_min) / w) * ((y_max - y_min) / h)
    total_points = n * Ti
    D = H / (total_points * bin_area)

    norm = LogNorm(vmin=max(D[D>0].min() if (D>0).any() else 1.0, 1e-12),
                   vmax=D.max() if D.size else 1.0) if log else None
    extent = [
        x_min if samplerate_hz is None else (x_min/samplerate_hz*1e3),
        x_max if samplerate_hz is None else (x_max/samplerate_hz*1e3),
        y_min, y_max
    ]
    ax.imshow(D, extent=extent, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.grid(color="1", alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Time (ms)" if samplerate_hz else "Sample")
    ax.set_facecolor("black")
    for s in ("top","right"): ax.spines[s].set_visible(False)

def cluster_activity_kde_ax(
    spike_times_ms, cluster_ids, recording_duration_ms,
    ax, time_pixels=100, cmap="inferno"
):
    """
    Plots the cluster activity KDE (occupancy heatmap) directly into the given axes `ax`.
    """
    spike_times_ms = np.asarray(spike_times_ms, dtype=np.float64)
    cluster_ids    = np.asarray(cluster_ids)

    clusters = np.sort(np.unique(cluster_ids))[::-1]
    K = clusters.size

    T = float(recording_duration_ms)
    m = (spike_times_ms >= 0) & (spike_times_ms < T)
    t_min = 0.0
    t_max = T / 60000.0  # minutes

    t_grid_min = np.linspace(t_min, t_max, time_pixels)
    kde_matrix = np.zeros((K, time_pixels), dtype=np.float32)
    cluster_labels = []
    for r, cid in enumerate(clusters):
        t = spike_times_ms[(cluster_ids == cid) & m] / 60000.0
        if t.size > 1:
            try:
                kde_values = gaussian_kde(t).pdf(t_grid_min)
            except np.linalg.LinAlgError:
                kde_values = np.zeros_like(t_grid_min)
                j = np.argmin(np.abs(t_grid_min - t.mean())) if t.size else 0
                kde_values[j] = 1.0
        elif t.size == 1:
            kde_values = np.zeros_like(t_grid_min)
            j = np.argmin(np.abs(t_grid_min - t[0]))
            kde_values[j] = 1.0
        else:
            kde_values = np.zeros_like(t_grid_min)
        kde_matrix[r, :] = kde_values
        cluster_labels.append(f'Cl: {cid}')

    xtick_fracs = [0.0, 1/3, 2/3, 1.0]
    xtick_positions = [int(round(frac * (time_pixels-1))) for frac in xtick_fracs]
    xtick_positions = sorted(set(xtick_positions))

    xticklabels = []
    for i in xtick_positions:
        label_val = t_grid_min[i]
        label = f"{label_val:.1f}"
        if np.isclose(label_val, round(label_val)):
            label = str(int(round(label_val)))
        xticklabels.append(label)

    full_xticklabels = [''] * time_pixels
    for pos, label in zip(xtick_positions, xticklabels):
        full_xticklabels[pos] = label

    sns.heatmap(
        kde_matrix,
        cmap=cmap,
        yticklabels=cluster_labels,
        xticklabels=full_xticklabels,
        ax=ax,
        cbar=False
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Presence Plot")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.invert_yaxis()
    ax.vlines(np.arange(0.5, time_pixels, 1), *ax.get_ylim(), color='k', linewidth=0.1, alpha=0.03)
    for pos in xtick_positions:
        ax.axvline(pos, *ax.get_ylim(), color="w", linestyle="--", linewidth=1, alpha=0.6, zorder=10)
    for y in range(1, K):
        ax.hlines(y, *ax.get_xlim(), color="k", linewidth=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def make_cluster_report(
    data,
    calc_metrics=True,
    metrics_df=None,
    SS=None,
    exclude_cluster_0=True,
    samplerate_hz=None,
    clusters_per_page=6,
    bin_duration_ms=60000.0,
    refractory_ms=3.0,
    n_neighbors=5,
    n_jobs=-1,
    rng_seed=42,
    max_waveforms_per_cluster=1000
):
    """
    Generate comprehensive cluster quality report from spike sorting data.
    
    Parameters
    ----------
    data : dict
        Dictionary from loadmat() containing:
        - 'cluster_class': (n_spikes, 2) array [cluster_id, spike_time_ms]
        - 'spikes': (n_spikes, n_samples) waveforms
        - 'inspk': (n_spikes, n_features) feature matrix
    calc_metrics : bool, optional
        If True, compute metrics internally. If False, use provided metrics_df and SS.
        Default: True
    metrics_df : pd.DataFrame, optional
        Pre-computed metrics DataFrame (required if calc_metrics=False)
    SS : np.ndarray, optional
        Pre-computed silhouette score matrix (required if calc_metrics=False)
    exclude_cluster_0 : bool, optional
        Whether to exclude cluster 0 (noise) from metrics computation. Default: True
    samplerate_hz : float, optional
        Sampling rate in Hz for time axis conversion. Default: None (use samples)
    clusters_per_page : int, optional
        Number of clusters to show per page. Default: 6
    bin_duration_ms : float, optional
        Bin duration for presence ratio in ms. Default: 60000.0 (1 minute)
    refractory_ms : float, optional
        Refractory period for ISI violations in ms. Default: 3.0
    n_neighbors : int, optional
        Number of neighbors for nearest neighbor metrics. Default: 5
    n_jobs : int, optional
        Number of parallel jobs for metrics computation. -1 uses all cores. Default: -1
    rng_seed : int, optional
        Random seed for reproducibility. Default: 42
    max_waveforms_per_cluster : int, optional
        Maximum number of waveforms to plot per cluster. Default: 1000
    
    Returns
    -------
    figs : list
        List of matplotlib figure objects
    metrics_df : pd.DataFrame
        DataFrame with all computed metrics
    SS : np.ndarray
        Silhouette score matrix
    """
    
    # Extract data
    cluster_class = data['cluster_class']
    waveforms = data['spikes']
    features = data['inspk']
    
    cluster_ids = cluster_class[:, 0].astype(int)
    spike_times_ms = cluster_class[:, 1]
    unique_clusters = np.unique(cluster_ids)
    recording_duration_ms = spike_times_ms.max()
    
    # ========================================================================
    # COMPUTE METRICS IF REQUESTED
    # ========================================================================
    if calc_metrics:
        print("Computing metrics...")
        metrics_df, SS = compute_cluster_metrics(data)
    
    else:
        if metrics_df is None or SS is None:
            raise ValueError("If calc_metrics=False, you must provide metrics_df and SS")
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    
    N, T = waveforms.shape
    clusters = np.unique(cluster_ids)
    K = len(clusters)
    
    leicolors_list = [
        "#000000", "#0000FF", "#ff0000ff", "#008000", "#9E0000", "#6A00C1",
        "#F68108", "#522500ff", "#FF1ABA", "#8B8B8B", "#96D34F", "#F69EDC", "#9BC1FF"
    ]
    leicolors = lambda x: leicolors_list[x % len(leicolors_list)]
    
    rng = np.random.default_rng(rng_seed)
    color_map = {c: leicolors(i) for i, c in enumerate(clusters)}
    
    pages = [clusters[i:i+clusters_per_page] for i in range(0, K, clusters_per_page)]
    figs = []
    
    per_col_width = 4.2
    per_fig_height = 11.5
    WSPACE = 0.15
    HSPACE = 0.25
    
    # Calculate actual max columns needed (1 summary column + max clusters on any page)
    max_clusters_on_page = max(len(page) for page in pages) if pages else 0
    ncols_per_page = 1 + min(clusters_per_page, max_clusters_on_page)
    consistent_figsize = (per_col_width * ncols_per_page, per_fig_height)
    
    # Create summary page
    summary_fig = None
    if len(pages) > 0:
        summary_fig = plt.figure(figsize=consistent_figsize, dpi=110)
        gs = summary_fig.add_gridspec(3, ncols_per_page, height_ratios=[1.0, 1.1, 0.9], 
                                      hspace=HSPACE, wspace=WSPACE)
        
        # Column 1A: Mean waveforms overlay
        axA = summary_fig.add_subplot(gs[0, 0])
        t = (np.arange(T)/samplerate_hz*1e3) if samplerate_hz else np.arange(T)
        total_spikes = len(cluster_ids)
        for i, c in enumerate(clusters):
            Wc = waveforms[cluster_ids == c]
            if Wc.size == 0: continue
            axA.plot(t, Wc.mean(axis=0), color=color_map[c], lw=2.2, label=f"C{c}")
        axA.set_title(f"Means (total n = {total_spikes})", fontsize=10, pad=6)
        axA.set_xlabel("Time (ms)" if samplerate_hz else "Sample")
        axA.set_ylabel("Amplitude")
        for s in ("top","right"): axA.spines[s].set_visible(False)
        axA.grid(alpha=0.25, linewidth=0.8)
        
        # Column 1B: Presence plot
        axB = summary_fig.add_subplot(gs[1, 0])
        cluster_activity_kde_ax(spike_times_ms, cluster_ids, recording_duration_ms, axB)
        
        # Column 1C: SNR bar plot
        axC = summary_fig.add_subplot(gs[2, 0])
        if isinstance(metrics_df, pd.DataFrame) and {'cluster_id','snr'}.issubset(metrics_df.columns):
            snr_map = dict(zip(metrics_df['cluster_id'].to_numpy(), metrics_df['snr'].to_numpy()))
            snrs = [snr_map.get(c, np.nan) for c in clusters]
            x = np.arange(len(clusters))
            bar_colors = [color_map[c] for c in clusters]
            axC.bar(x, snrs, color=bar_colors, width=0.9)
            axC.set_xticks(x, [f"C {c}" for c in clusters], rotation=90)
            axC.set_ylabel("SNR")
            axC.set_title("SNR by cluster", fontsize=10, pad=6)
            for s in ("top","right"): axC.spines[s].set_visible(False)
            axC.grid(axis="y", alpha=0.25, linewidth=0.8)
        else:
            axC.text(0.5, 0.5, "metrics_df missing\n('cluster_id','snr')",
                     ha="center", va="center", fontsize=10)
            axC.set_axis_off()
    
    # Generate pages
    for page_i, subset in enumerate(pages, start=1):
        if page_i == 1 and 0 in subset:
            subset = [c for c in subset if c != 0] + [0]
        
        if page_i == 1:
            fig = summary_fig
            gs = fig.axes[0].get_gridspec() if len(fig.axes) > 0 else fig.add_gridspec(
                3, ncols_per_page, height_ratios=[1.0,1.1,0.9], hspace=HSPACE, wspace=WSPACE
            )
            cluster_col_start = 1
        else:
            fig = plt.figure(figsize=consistent_figsize, dpi=110)
            gs = fig.add_gridspec(3, ncols_per_page, height_ratios=[1.0, 1.1, 0.9], 
                                 hspace=HSPACE, wspace=WSPACE)
            cluster_col_start = 1
        
        # Per-cluster columns
        for j, cid in enumerate(subset, start=cluster_col_start):
            ax1 = fig.add_subplot(gs[0, j])  # raw + mean
            ax2 = fig.add_subplot(gs[1, j])  # density
            ax3 = fig.add_subplot(gs[2, j])  # ISI
            
            mask = (cluster_ids == cid)
            W = waveforms[mask]
            times = spike_times_ms[mask]
            
            if W.size == 0:
                for a in (ax1, ax2, ax3): a.set_axis_off()
                continue
            
            t = (np.arange(T)/samplerate_hz*1e3) if samplerate_hz else np.arange(T)
            n_here = W.shape[0]
            take = min(max_waveforms_per_cluster, n_here)
            pick = rng.choice(n_here, size=take, replace=False) if n_here > take else np.arange(n_here)
            ax1.plot(t, W[pick].T, color=color_map[cid], alpha=0.15, lw=0.8)
            
            mean_waveform = W.mean(axis=0)
            ax1.plot(t, mean_waveform, color='k', lw=2.4, label='Mean')
            
            ax1.set_title(f"Cluster {cid} (n={n_here})", fontsize=10, pad=6)
            ax1.set_xlabel("Time (ms)" if samplerate_hz else "Sample")
            for s in ("top","right"): ax1.spines[s].set_visible(False)
            ax1.grid(alpha=0.25, linewidth=0.8)
            
            # Density
            _density_image(W, ax2, samplerate_hz=samplerate_hz)
            
            # ISI
            times_sorted = np.sort(times)
            isi_ms = np.diff(times_sorted)
            isi_ms = isi_ms[(isi_ms >= 0) & np.isfinite(isi_ms)]
            
            nbins = 50
            bin_step = 2
            bins = np.arange(0, bin_step * nbins + bin_step, bin_step)
            
            ax3.autoscale(False, axis='x')
            ax3.hist(isi_ms, bins=bins, rwidth=1, linewidth=0, color=color_map[cid])
            
            n_viol = np.count_nonzero(isi_ms < refractory_ms)
            ax3.set_title(f"{n_viol} in < {refractory_ms:.0f}ms", fontsize=10, pad=6)
            ax3.set_xlabel("ISI (ms)")
            ax3.set_xlim([0, bin_step * nbins])
            for s in ("top","right"): ax3.spines[s].set_visible(False)
            ax3.grid(alpha=0.25, linewidth=0.8)
        
        # Fill empty columns
        for pad_col in range(cluster_col_start + len(subset), ncols_per_page):
            for row in range(3):
                ax_pad = fig.add_subplot(gs[row, pad_col])
                ax_pad.set_axis_off()
        
        fig.suptitle(f"Cluster Report — Page {page_i}/{len(pages)}", fontsize=12, y=0.995)
        fig.tight_layout(rect=(0.01, 0, 1, 0.99))
        figs.append(fig)
        plt.show()
    
    # Metrics summary page
    if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
        cluster_ids_all = metrics_df['cluster_id'].to_numpy() if 'cluster_id' in metrics_df.columns else np.array([])
        metrics_cols = [col for col in metrics_df.columns
                        if col not in ('cluster_id', 'snr', 'SNR', 'presence_ratio', 
                                      'presence ratio', 'PresenceRatio','num_spikes',)]
        n_metrics = len(metrics_cols)
        # Use fixed layout independent of clusters_per_page
        # Reserve space for metrics + silhouette heatmap
        cols_per_row = 6  # Fixed reasonable number for metrics layout
        n_rows = max(1, (n_metrics + 1 + cols_per_row - 1) // cols_per_row)  # +1 for silhouette heatmap
        
        if n_metrics > 0 and cluster_ids_all.size > 0:
            fig_metrics, axes = plt.subplots(
                nrows=n_rows, ncols=cols_per_row,
                figsize=(4.0 * cols_per_row, 4.0 * n_rows),
                dpi=300, sharex=False
            )
            
            plt.subplots_adjust(wspace=0.07, hspace=0.13)
            
            axes = np.atleast_2d(axes)
            axes_flat = axes.flatten()
            
            for i, col in enumerate(metrics_cols):
                ax = axes_flat[i]
                ys = metrics_df[col].to_numpy()
                bar_colors = [leicolors(idx+1) for idx in range(len(cluster_ids_all))]
                x = np.arange(len(cluster_ids_all))
                ax.bar(x, ys, color=bar_colors, width=0.9)
                if i // cols_per_row == n_rows - 1:
                    xticks_labels = [f"c{c}" for c in cluster_ids_all]
                    ax.set_xticks(x, xticks_labels, rotation='vertical')
                else:
                    ax.set_xticks([])
                ax.set_ylabel(col)
                ax.set_title(col, fontsize=11, pad=6)
                ax.grid(axis="y", alpha=0.3, linewidth=0.8)
                for s in ("top","right"):
                    ax.spines[s].set_visible(False)
            
            # Silhouette heatmap - always plot it after metrics
            sil_idx = len(metrics_cols)
            if sil_idx < len(axes_flat) and SS is not None:
                ax_sil = axes_flat[sil_idx]
                S = np.where(np.isnan(SS), SS.T, SS)
                np.fill_diagonal(S, np.nan)
                mask = np.triu(np.ones_like(S, dtype=bool))
                unique_clusters_valid = cluster_ids_all if cluster_ids_all.size == S.shape[0] else range(S.shape[0])
                cbar_kws = {'shrink': 0.6, 'pad': 0.03, 'aspect': 10, 'label': 'Silhouette\nScore'}
                sns.heatmap(
                    S, annot=False, cmap='viridis', vmin=-1, vmax=1, center=0, mask=mask,
                    xticklabels=[f"c{c}" for c in unique_clusters_valid],
                    yticklabels=[f"c{c}" for c in unique_clusters_valid],
                    ax=ax_sil, cbar=True, cbar_kws=cbar_kws, square=True
                )
                ax_sil.set_title('Silhouette Score\nHeatmap', fontsize=10)
                ax_sil.set_xlabel('Cluster')
                ax_sil.set_ylabel('Cluster')
                ax_sil.tick_params(axis='x', labelrotation=90, labelsize=7)
                ax_sil.tick_params(axis='y', labelsize=7)
                for s in ("top","right"):
                    ax_sil.spines[s].set_visible(False)
                sil_plotted = True
            else:
                sil_plotted = False
            
            # Turn off remaining empty axes
            start_idx = len(metrics_cols) + (1 if sil_plotted else 0)
            for j in range(start_idx, n_rows * cols_per_row):
                axes_flat[j].set_axis_off()
            
            fig_metrics.suptitle("Cluster Metrics Summary", fontsize=13, y=1.04)
            fig_metrics.tight_layout(rect=(0.01, 0, 1, 0.97))
            figs.append(fig_metrics)
            plt.show()
    
    return figs, metrics_df, SS