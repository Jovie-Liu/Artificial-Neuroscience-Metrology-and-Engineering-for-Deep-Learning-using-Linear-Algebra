# ════════════════════════════════════════════════════════════════════════════════
# SVD Visualization for Linear Layers
# ════════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union, Literal, Tuple, Optional, Any

def parse_layer_info(weight_name: str) -> Dict[str, Any]:
    """
    Parse layer information from weight name.
    Args: weight_name: Full weight name 
    Returns: Dictionary with parsed information
    """
    info = {
        'layer_idx': None,
        'layer_type': None,
        'is_conv_layer': False,
        'full_name': weight_name
    }
    
    # Check if it's a conv_layer
    conv_layer_match = re.search(r'conv_layers\.(\d+)\.conv_(\w+)\.weight', weight_name)
    if conv_layer_match:
        info['layer_idx'] = int(conv_layer_match.group(1))
        info['layer_type'] = conv_layer_match.group(2)  # expand, depthwise, residual, skip
        info['is_conv_layer'] = True
        return info
    
    # Check for other known layers
    if 'encoder.weight' in weight_name:
        info['layer_type'] = 'encoder'
    elif 'decoder.weight' in weight_name:
        info['layer_type'] = 'decoder'
    elif 'bottleneck.weight' in weight_name:
        info['layer_type'] = 'bottleneck'
    elif 'output_layer' in weight_name:
        info['layer_type'] = 'output'
    
    return info

def determine_grid_layout(num_layer_types: int) -> Tuple[int, int]:
    """
    Determine grid layout based on the number of layer types.
    
    Args:
        num_layer_types: Number of distinct layer types in conv_layers
        
    Returns:
        (num_cols, rows_per_index)
        - num_cols: Number of columns in the grid
        - rows_per_index: How many rows each layer index occupies
    """
    if num_layer_types == 2:
        # 2 types → 4 columns, but we'll handle 2 indices per row specially
        return 4, 1
    elif num_layer_types == 3:
        # 3 types → 3 columns, 1 row per index
        return 3, 1
    elif num_layer_types == 4:
        # 4 types → 4 columns, 1 row per index
        return 4, 1
    elif num_layer_types == 5:
        # 5 types → 3 columns, 2 rows per index (last subplot blank)
        return 3, 2
    elif num_layer_types == 6:
        # 6 types → 3 columns, 2 rows per index
        return 3, 2
    elif num_layer_types == 7:
        # 7 types → 4 columns, 2 rows per index (last subplot blank)
        return 4, 2
    elif num_layer_types == 8:
        # 8 types → 4 columns, 2 rows per index
        return 4, 2
    elif num_layer_types == 9:
        # 9 types → 3 columns, 3 rows per index
        return 3, 3
    else:
        # Default: 4 columns, compute rows needed
        rows_per_index = (num_layer_types + 3) // 4
        return 4, rows_per_index
    
def organize_weights_for_plotting(
    evolutions: Dict[str, Dict]
) -> Tuple[List[str], Dict[int, Dict[str, str]], List[str], int, int]:
    """
    Organize weights into non-conv-layer and conv-layer groups.
    
    Args:
        evolutions: Output from analyze_weight_evolution
        
    Returns:
        - non_conv_weights: List of non-conv-layer weight names
        - conv_weights_by_idx: Dict mapping layer_idx -> {layer_type: weight_name}
        - layer_type_order: Sorted list of layer types
        - num_cols: Number of columns for the grid
        - rows_per_index: Rows per layer index
    """
    non_conv_weights = []
    conv_weights_by_idx = defaultdict(dict)
    layer_types = set()
    
    for weight_name in evolutions.keys():
        info = parse_layer_info(weight_name)
        
        if info['is_conv_layer']:
            layer_idx = info['layer_idx']
            layer_type = info['layer_type']
            conv_weights_by_idx[layer_idx][layer_type] = weight_name
            layer_types.add(layer_type)
        else:
            non_conv_weights.append(weight_name)
    
    # Sort layer types alphabetically for consistent ordering
    layer_type_order = sorted(layer_types)
    
    # Determine grid layout
    num_layer_types = len(layer_type_order)
    num_cols, rows_per_index = determine_grid_layout(num_layer_types)
    
    return non_conv_weights, dict(conv_weights_by_idx), layer_type_order, num_cols, rows_per_index

# ─────────────────────────────────────────────────────────────────────────── #
# 1. PLOT SPECIFICATION
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class PlotSpec:
    """
    Bundles the drawing callable and its configuration for visualize_evolution_grid.

    The layout engine calls plot_spec.fn once per weight slot with this signature:

        fn(
            axes_slice,   # List[plt.Axes]  length = col_multiplier
            evolution,    # Dict            evolutions[weight_name]
            title,        # str
            show_xlabel,  # bool
            show_ylabel,  # bool
            **kwargs,     # forwarded from PlotSpec.kwargs
        ) -> None

    Single-column callables use axes_slice[0] only.
    Dual-column callables (e.g. angular 'both') use axes_slice[0] and axes_slice[1].

    Attributes:
        fn:                  Callable with the signature above.
        kwargs:              Extra keyword arguments forwarded verbatim to fn.
        col_multiplier:      Columns consumed per weight slot.
                             1 for all scalar/SV plots.
                             2 for angular 'both' (left + right side by side).
        figure_title:        Figure suptitle.
        legend_title:        Shared legend title. Pass None to suppress.
        figsize_per_subplot: (width, height) per subplot cell in inches.
    """
    fn:                  Callable
    kwargs:              Dict[str, Any]       = field(default_factory=dict)
    col_multiplier:      int                  = 1
    figure_title:        str                  = 'Weight Matrix Evolution'
    legend_title:        Optional[str]        = 'Legend'
    figsize_per_subplot: Tuple[float, float]  = (3.5, 2.5)

# ─────────────────────────────────────────────────────────────────────────── #
# 2. MANDATORY CALLABLE SIGNATURE:
#
#   fn(axes_slice, evolution, title, show_xlabel, show_ylabel, **kwargs)
#
# axes_slice : List[plt.Axes]  — length equals PlotSpec.col_multiplier
# evolution  : Dict            — one entry from evolutions[weight_name]
# title      : str
# show_xlabel: bool
# show_ylabel: bool
# ─────────────────────────────────────────────────────────────────────────── #

def plot_singular_values_single(
    ax: plt.Axes,
    evolution: Dict,
    title: str,
    top_k_to_show: int = 5,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_effective_rank: bool = True,   # ← NEW
    consistent_coloring: bool = True,
) -> None:
    """
    Plot singular value evolution on a single axes.
    Optionally draws the effective-rank interpolated singular value as a single overlay curve on the same y-axis.

    Args:
        ax:                  Matplotlib axes.
        evolution:           Evolution data for a single weight matrix.
        top_k_to_show:       Number of top singular value curves to draw.
        show_xlabel:         Whether to label the x-axis.
        show_ylabel:         Whether to label the y-axis.
        show_effective_rank: Overlay the σ(effective_rank) line.
                             Silently skipped if 'effective_rank_sv' is absent.
    """
    epochs               = evolution['epochs']
    singular_values_list = evolution['singular_values']

    # ------------------------------------------------------------------ #
    # Build (num_epochs, k_stored) matrix
    # ------------------------------------------------------------------ #
    sv_matrix = []
    for sv in singular_values_list:
        sv_np = (sv.detach().cpu().numpy()
                 if isinstance(sv, torch.Tensor)
                 else np.array(sv))
        sv_matrix.append(sv_np)

    sv_matrix = np.stack(sv_matrix, axis=0)   # (num_epochs, k_stored)
    k_plot    = min(top_k_to_show, sv_matrix.shape[1])
    colors    = plt.cm.viridis(np.linspace(0, 0.9, k_plot))

    # ------------------------------------------------------------------ #
    # Singular value curves
    # ------------------------------------------------------------------ #
    for i in range(k_plot):
        if consistent_coloring:
            ax.plot(
                epochs,
                sv_matrix[:, i],
                color     = colors[i],
                linewidth = 1.5,
                label     = f'$\\sigma_{{{i+1}}}$',
                alpha     = 0.8,
                zorder    = 3,
            )
        else:
            ax.plot(epochs, sv_matrix[:, i], 'o-', label = f'$\\sigma_{{{i+1}}}$',linewidth = 1.6)

    # ------------------------------------------------------------------ #
    # Effective-rank SV overlay  (precomputed, same y-axis)
    # ------------------------------------------------------------------ #
    
    if show_effective_rank and 'effective_rank_sv' in evolution:

        er_sv_raw = evolution['effective_rank_sv']

        # Convert list of scalar tensors / floats to numpy
        er_sv = np.array([
            v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for v in er_sv_raw
        ])

        # Build matched (epoch, y) pairs, dropping NaN entries
        valid_mask   = ~np.isnan(er_sv)
        valid_epochs = [ep for ep, m in zip(epochs, valid_mask) if m]
        valid_er_sv  = er_sv[valid_mask]

        nan_count = int((~valid_mask).sum())
        if nan_count > 0:
            print(
                f"  [{title}] Note: effective_rank_sv is NaN at {nan_count} "
                f"epoch(s) — effective rank exceeded the full spectrum length."
            )

        if len(valid_epochs) > 0:
            ax.plot(
                valid_epochs,
                valid_er_sv,
                color     = '#E8735A',      # coral, distinct from viridis
                linewidth = 2.0,
                linestyle = '--',
                # marker    = 'D',            # diamond
                markersize= 3.5,
                alpha     = 0.90,
                zorder    = 5,
                label     = r'$\sigma$(eff. rank)',
            )

    # ------------------------------------------------------------------ #
    # Axes formatting
    # ------------------------------------------------------------------ #
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(min(epochs), max(epochs))

    if show_xlabel:
        ax.set_xlabel('Epoch', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Singular Value', fontsize=8)

    ax.tick_params(axis='both', labelsize=7)

def sv_plot_fn(
    axes_slice:  List[plt.Axes],
    evolution:   Dict,
    title:       str,
    show_xlabel: bool,
    show_ylabel: bool,
    top_k_to_show:       int  = 10,
    show_effective_rank: bool = False,
    consisistent_coloring: bool = True,
) -> None:
    """Singular value evolution. Wraps plot_singular_values_single."""
    plot_singular_values_single(
        ax                  = axes_slice[0],
        evolution           = evolution,
        title               = title,
        top_k_to_show       = top_k_to_show,
        show_xlabel         = show_xlabel,
        show_ylabel         = show_ylabel,
        show_effective_rank = show_effective_rank,
        consistent_coloring = consisistent_coloring,
    )


def effective_rank_plot_fn(
    axes_slice:  List[plt.Axes],
    evolution:   Dict,
    title:       str,
    show_xlabel: bool,
    show_ylabel: bool,
    color:      str   = '#E8735A',
    marker:     str   = 'o',
    markersize: float = 3.0,
) -> None:
    """
    Effective rank over epochs.
    Reads: evolution['epochs'], evolution['effective_ranks']
    """
    ax     = axes_slice[0]
    epochs = evolution['epochs']
    er     = np.array([
        v.item() if isinstance(v, torch.Tensor) else float(v)
        for v in evolution['effective_ranks']
    ])

    ax.plot(epochs, er, color=color, linewidth=1.6,
            marker=marker, markersize=markersize, alpha=0.88)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=7)
    if show_xlabel:
        ax.set_xlabel('Epoch', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Effective Rank', fontsize=8)


def condition_number_plot_fn(
    axes_slice:  List[plt.Axes],
    evolution:   Dict,
    title:       str,
    show_xlabel: bool,
    show_ylabel: bool,
    color:     str  = '#4A90D9',
    marker:    str  = 's',
    markersize: float = 3.0,
    log_scale: bool = False,
) -> None:
    """
    Condition number over epochs. Auto log-scale when range > 2 decades.
    Reads: evolution['epochs'], evolution['condition_numbers']
    """
    ax     = axes_slice[0]
    epochs = evolution['epochs']
    cn     = np.array([
        v.item() if isinstance(v, torch.Tensor) else float(v)
        for v in evolution['condition_numbers']
    ])

    ax.plot(epochs, cn, color=color, linewidth=1.6,
            marker=marker, markersize=markersize, alpha=0.88)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlim(min(epochs), max(epochs))
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=7)

    # Auto log-scale when range spans > 2 decades
    cn_min = np.nanmin(cn)
    cn_max = np.nanmax(cn)
    if log_scale and cn_max / (cn_min + 1e-12) > 100:
        ax.set_yscale('log')

    if show_xlabel:
        ax.set_xlabel('Epoch', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Condition Number', fontsize=8)

def individual_angles_plot_fn(
    axes_slice:  List[plt.Axes],
    evolution:   Dict,
    title:       str,
    show_xlabel: bool,
    show_ylabel: bool,
    # ── kwargs ────────────────────────────────────────────────────────── #
    vector_type:         str   = 'both',  # 'left' | 'right' | 'both'
    top_k_to_show:       int   = 5,
    show_threshold_line: bool  = True,
    consistent_coloring: bool  = True,
) -> None:
    """
    Individual singular vector angular changes between consecutive epochs.

    vector_type='left'  -> axes_slice[0] only   (left vectors U)
    vector_type='right' -> axes_slice[0] only   (right vectors V)
    vector_type='both'  -> axes_slice[0]=U, axes_slice[1]=V
                           requires col_multiplier=2 in PlotSpec

    x-axis: epochs[1:]
    Reads: evolution['epochs'], evolution['left/right_angles']
    """
    x = np.array(evolution['epochs'][1:])

    def _draw(ax, angle_data, vec_label, show_ylabel_flag):
        if angle_data is None or angle_data.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(title, fontsize=9, fontweight='bold')
            return
        
        k_plot = min(top_k_to_show, angle_data.shape[1])
        colors = plt.cm.plasma(np.linspace(0.15, 0.85, k_plot))

        for i in range(k_plot):
            if consistent_coloring:
                ax.plot(x, angle_data[:, i],
                        color=colors[i], linewidth=1.6,
                        marker='o', markersize=2.8, alpha=0.85,
                        label=f'${vec_label}_{{{i+1}}}$')
            else:
                ax.plot(x, angle_data[:, i], 'o-', label=f'${vec_label}_{{{i+1}}}$', linewidth=1.6)

        if show_threshold_line:
            ax.axhline(5.0, color='#AAAAAA', linewidth=0.8,
                       linestyle=':', zorder=0)

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if show_xlabel:
            ax.set_xlabel('Epoch', fontsize=8)
        if show_ylabel_flag:
            ax.set_ylabel('Angular Change (deg)', fontsize=8)

    if vector_type == 'left':
        _draw(axes_slice[0], evolution['left_angles'],  'u', show_ylabel)

    elif vector_type == 'right':
        _draw(axes_slice[0], evolution['right_angles'], 'v', show_ylabel)

    else:  # 'both'
        _draw(axes_slice[0], evolution['left_angles'],  'u', show_ylabel)
        _draw(axes_slice[1], evolution['right_angles'], 'v', False)
        # Disambiguate titles when both panels share the same weight title
        axes_slice[0].set_title(f'{title} — U', fontsize=9, fontweight='bold')
        axes_slice[1].set_title(f'{title} — V', fontsize=9, fontweight='bold')


def principal_angles_plot_fn(
    axes_slice:  List[plt.Axes],
    evolution:   Dict,
    title:       str,
    show_xlabel: bool,
    show_ylabel: bool,
    vector_type:   str  = 'both',   # 'left' | 'right' | 'both'
    top_k_to_show: int  = 5,
    consistent_coloring: bool = True,
) -> None:
    """
    Principal angles between consecutive epoch subspaces.

    Identical layout logic to individual_angles_plot_fn but reads
    left/right_principal_angles instead of left/right_angles.
    vector_type='both' requires col_multiplier=2 in PlotSpec.

    x-axis: epochs[1:]
    Reads: evolution['epochs'], evolution['left/right_principal_angles']
    """
    x = np.array(evolution['epochs'][1:])

    def _draw(ax, angle_data, vec_label, show_ylabel_flag):
        if angle_data is None or angle_data.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(title, fontsize=9, fontweight='bold')
            return

        k_plot = min(top_k_to_show, angle_data.shape[1])
        colors  = plt.cm.viridis(np.linspace(0.15, 0.85, k_plot))
        
        for i in range(k_plot):
            if consistent_coloring:
                ax.plot(x, angle_data[:, i],
                        color=colors[i], linewidth=1.6,
                        marker='s', markersize=2.8, alpha=0.85,
                        label=f'$\\theta_{{{i+1}}}$')
            else:
                ax.plot(x, angle_data[:, i], 'o-', label=f'${vec_label}_{{{i+1}}}$', linewidth=1.6)

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 90)           # principal angles always in [0, 90] degrees
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if show_xlabel:
            ax.set_xlabel('Epoch', fontsize=8)
        if show_ylabel_flag:
            ax.set_ylabel('Principal Angle (deg)', fontsize=8)

    if vector_type == 'left':
        _draw(axes_slice[0], evolution['left_principal_angles'],  'u', show_ylabel)

    elif vector_type == 'right':
        _draw(axes_slice[0], evolution['right_principal_angles'], 'v', show_ylabel)

    else:  # 'both'
        _draw(axes_slice[0], evolution['left_principal_angles'],  'u', show_ylabel)
        _draw(axes_slice[1], evolution['right_principal_angles'], 'v', False)
        axes_slice[0].set_title(f'{title} — U', fontsize=9, fontweight='bold')
        axes_slice[1].set_title(f'{title} — V', fontsize=9, fontweight='bold')

def grassmann_dist_plot_fn(
    axes_slice:  List[plt.Axes],
    evolution:   Dict,
    title:       str,
    show_xlabel: bool,
    show_ylabel: bool,
    vector_type: str   = 'both',   # 'left' | 'right' | 'both'
    color_left:  str   = '#E8735A',
    color_right: str   = '#4A90D9',
    marker:      str   = 'o',
    markersize:  float = 3.0
) -> None:
    """
    Grassmann geodesic distance between consecutive epoch subspaces.

    When vector_type='both', overlays left and right on the same axes.
    Reads: evolution['epochs'], evolution['left/right_grassmann_dist']
    x-axis: epochs[1:]  (the epoch at which the new subspace was observed)
    """
    ax     = axes_slice[0]
    # Transition x-axis: epoch at which the NEW state lands
    x      = np.array(evolution['epochs'][1:])

    if vector_type in ('left', 'both'):
        dist = evolution['left_grassmann_dist']       # ndarray (n-1,)
        ax.plot(x, dist, color=color_left, linewidth=1.6,
                marker=marker, markersize=markersize,
                alpha=0.88, label='Left (U)')

    if vector_type in ('right', 'both'):
        dist = evolution['right_grassmann_dist']      # ndarray (n-1,)
        ax.plot(x, dist, color=color_right, linewidth=1.6,
                marker=marker, markersize=markersize,
                alpha=0.88, label='Right (V)')

    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=7)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if vector_type == 'both':
        ax.legend(fontsize=7, loc='upper right')
    if show_xlabel:
        ax.set_xlabel('Epoch', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Grassmann Distance (rad)', fontsize=8)

def sv_spectrum_plot_fn(
    axes_slice:  List[plt.Axes],
    evolution:   Dict,
    title:       str,
    show_xlabel: bool,
    show_ylabel: bool,
    epoch_idx:   int  = -1,    # which epoch snapshot to show; -1 = latest
    log_scale:   bool = False,
) -> None:
    """
    Singular value spectrum at a single epoch snapshot (SV vs. rank index).

    Unlike sv_plot_fn which traces each SV across training epochs, this
    plots the full stored spectrum at one point in time — i.e. the classic
    'scree plot' view of a weight matrix.

    x-axis : singular value rank index  (1 .. k)
    y-axis : singular value magnitude

    Reads from evolution:
        'singular_values'   List[Tensor (k,)]  — indexed by epoch_idx
        'effective_ranks'   List[scalar]       — for the vertical marker
        'condition_numbers' List[scalar]       — for the stats annotation
        'epochs'            List[int]          — for the subplot subtitle

    Args:
        epoch_idx:  Index into the epoch list. Defaults to -1 (last epoch).
        log_scale:  Apply log scale to the y-axis.
    """
    ax = axes_slice[0]

    # ── Pull the snapshot ──────────────────────────────────────────────── #
    sv      = evolution['singular_values'][epoch_idx]       # Tensor (k,)
    sv      = sv.cpu().numpy() if isinstance(sv, torch.Tensor) else np.array(sv)

    eff_rank = evolution['effective_ranks'][epoch_idx]
    eff_rank = eff_rank.item() if isinstance(eff_rank, torch.Tensor) else float(eff_rank)

    cond_num = evolution['condition_numbers'][epoch_idx]
    cond_num = cond_num.item() if isinstance(cond_num, torch.Tensor) else float(cond_num)

    epoch_label = evolution['epochs'][epoch_idx]

    # ── Plot ──────────────────────────────────────────────────────────── #
    indices = np.arange(1, len(sv) + 1)

    ax.plot(indices, sv, 'o-', color='steelblue', linewidth=2, markersize=4)
    ax.axvline(eff_rank, color='red', linestyle='--',
               linewidth=1.2, label=f'Eff. Rank: {eff_rank:.2f}')

    # ── Stats annotation ──────────────────────────────────────────────── #
    textstr = (
        f'Max:  {sv[0]:.2e}\n'
        f'Min:  {sv[-1]:.2e}\n'
        f'Cond: {cond_num:.2e}'
    )
    ax.text(
        0.98, 0.97, textstr,
        transform            = ax.transAxes,
        verticalalignment    = 'top',
        horizontalalignment  = 'right',
        bbox                 = dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize             = 8,
    )

    # ── Formatting ────────────────────────────────────────────────────── #
    ax.set_title(f'{title}\n(epoch {epoch_label})', fontsize=9, fontweight='bold')
    ax.set_xlim(1, len(sv))
    ax.set_ylim(bottom=0 if not log_scale else None)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=7)
    ax.legend(fontsize=7, loc='lower left')

    if log_scale:
        ax.set_yscale('log')

    if show_xlabel:
        ax.set_xlabel('Rank Index', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Singular Value', fontsize=8)


# ─────────────────────────────────────────────────────────────────────────── #
# 3. MAIN GRID LAYOUT ENGINE
# ─────────────────────────────────────────────────────────────────────────── #

def visualize_evolution_grid(
    evolutions: Dict[str, Dict],
    plot_spec:  PlotSpec,
    save_path:  Optional[str] = None,
    dpi:        int           = 150,
) -> plt.Figure:
    """
    Generic layout engine. Organises weight subplots with the same geometry
    as visualize_sv_evolution and delegates all drawing to plot_spec.fn.

    The callable in plot_spec.fn is invoked once per weight slot as:

        plot_spec.fn(
            axes_slice,          # List[plt.Axes]  len = col_multiplier
            evolution,           # evolutions[weight_name]
            title,               # str
            show_xlabel,         # bool
            show_ylabel,         # bool
            **plot_spec.kwargs,
        )

    Args:
        evolutions: Output from analyze_weight_evolution (all stats embedded).
        plot_spec:  PlotSpec describing what and how to draw.
        save_path:  Optional file path to save the figure.
        dpi:        Resolution for the saved figure.

    Returns:
        Matplotlib Figure.
    """
    cm = plot_spec.col_multiplier

    # ------------------------------------------------------------------ #
    # 1. Layout arithmetic
    # ------------------------------------------------------------------ #
    (non_conv_weights,
     conv_weights_by_idx,
     layer_type_order,
     base_num_cols,
     rows_per_index) = organize_weights_for_plotting(evolutions)

    num_cols             = base_num_cols * cm
    num_layer_types      = len(layer_type_order)
    sorted_layer_indices = sorted(conv_weights_by_idx.keys())

    non_conv_rows = (
        len(non_conv_weights) // base_num_cols + 1
        if len(non_conv_weights) > 0 else 0
    )

    if num_layer_types == 2:
        conv_rows = (len(sorted_layer_indices) + 1) // 2
    else:
        conv_rows = len(sorted_layer_indices) * rows_per_index

    total_rows = non_conv_rows + conv_rows

    # ------------------------------------------------------------------ #
    # 2. Create figure
    # ------------------------------------------------------------------ #
    w, h = plot_spec.figsize_per_subplot
    fig, axes = plt.subplots(
        total_rows, num_cols,
        figsize = (num_cols * w, total_rows * h),
        squeeze = False,
    )

    # ------------------------------------------------------------------ #
    # 3. Core dispatch helpers
    # ------------------------------------------------------------------ #
    def _dispatch(row, base_col, weight_name, title, show_xlabel, show_ylabel):
        axes_slice = [axes[row, base_col + offset] for offset in range(cm)]
        plot_spec.fn(
            axes_slice,
            evolutions[weight_name],
            title,
            show_xlabel,
            show_ylabel,
            **plot_spec.kwargs,
        )

    def _hide(row, base_col):
        for offset in range(cm):
            axes[row, base_col + offset].axis('off')

    current_row = 0

    # ================================================================
    # PART 1: Non-conv weights
    # ================================================================
    if len(non_conv_weights) > 0:
        non_conv_sorted = sorted(non_conv_weights)

        for flat_idx, weight_name in enumerate(non_conv_sorted):
            row_offset = flat_idx // base_num_cols
            base_col   = (flat_idx  % base_num_cols) * cm

            info  = parse_layer_info(weight_name)
            title = (info['layer_type'].capitalize()
                     if info['layer_type']
                     else weight_name.split('.')[-2])

            _dispatch(
                row        = current_row + row_offset,
                base_col   = base_col,
                weight_name= weight_name,
                title      = title,
                show_xlabel= (current_row + row_offset == total_rows - 1),
                show_ylabel= (base_col == 0),
            )

        # Hide unused slots
        for empty in range(len(non_conv_sorted), non_conv_rows * base_num_cols):
            _hide(current_row + empty // base_num_cols,
                  (empty  % base_num_cols) * cm)

        current_row += non_conv_rows

    # ================================================================
    # PART 2A: Conv weights — 2 layer types (2 indices per row)
    # ================================================================
    if num_layer_types == 2:
        for i in range(0, len(sorted_layer_indices), 2):
            for j in range(2):
                if i + j >= len(sorted_layer_indices):
                    for type_idx in range(len(layer_type_order)):
                        _hide(current_row, (j * len(layer_type_order) + type_idx) * cm)
                    continue

                layer_idx     = sorted_layer_indices[i + j]
                layer_weights = conv_weights_by_idx[layer_idx]

                for type_idx, layer_type in enumerate(layer_type_order):
                    base_col = (j * len(layer_type_order) + type_idx) * cm

                    if layer_type in layer_weights:
                        _dispatch(
                            row        = current_row,
                            base_col   = base_col,
                            weight_name= layer_weights[layer_type],
                            title      = f'Layer {layer_idx} - {layer_type.capitalize()}',
                            show_xlabel= (current_row == total_rows - 1),
                            show_ylabel= (base_col == 0),
                        )
                    else:
                        _hide(current_row, base_col)

            current_row += 1

    # ================================================================
    # PART 2B: Conv weights — general case
    # ================================================================
    else:
        for layer_idx in sorted_layer_indices:
            layer_weights = conv_weights_by_idx[layer_idx]
            type_counter  = 0

            for row_offset in range(rows_per_index):
                for col_slot in range(base_num_cols):
                    base_col = col_slot * cm

                    if type_counter >= len(layer_type_order):
                        _hide(current_row + row_offset, base_col)
                        continue

                    layer_type   = layer_type_order[type_counter]
                    type_counter += 1

                    if layer_type in layer_weights:
                        _dispatch(
                            row        = current_row + row_offset,
                            base_col   = base_col,
                            weight_name= layer_weights[layer_type],
                            title      = f'Layer {layer_idx} - {layer_type.capitalize()}',
                            show_xlabel= (current_row + row_offset == total_rows - 1),
                            show_ylabel= (base_col == 0),
                        )
                    else:
                        _hide(current_row + row_offset, base_col)

            current_row += rows_per_index

    # ================================================================
    # Shared legend + suptitle
    # ================================================================
    if plot_spec.legend_title is not None:
        for row in range(total_rows):
            for col in range(num_cols):
                handles, labels = axes[row, col].get_legend_handles_labels()
                if handles:
                    fig.legend(
                        handles, labels,
                        loc            = 'upper right',
                        bbox_to_anchor = (0.99, 0.99),
                        fontsize       = 8,
                        title          = plot_spec.legend_title,
                        title_fontsize = 9,
                        framealpha     = 0.9,
                    )
                    break
            else:
                continue
            break

    fig.suptitle(
        plot_spec.figure_title,
        fontsize   = 14,
        fontweight = 'bold',
        y          = 1.01,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Figure saved to: {save_path}')

    return fig

# ─────────────────────────────────────────────────────────────────────────── #
# 4. THIN WRAPPERS
# ─────────────────────────────────────────────────────────────────────────── #

def visualize_sv_evolution(
    evolutions:          Dict[str, Dict],
    top_k_to_show:       int                 = 10,
    show_effective_rank: bool                = False,
    figsize_per_subplot: Tuple[float, float] = (3.5, 2.5),
    legend_title:        Optional[str]       = 'Singular Values',
    save_path:           Optional[str]       = None,
    dpi:                 int                 = 150,
) -> plt.Figure:
    return visualize_evolution_grid(evolutions, PlotSpec(
        fn                  = sv_plot_fn,
        kwargs              = dict(top_k_to_show=top_k_to_show,
                                   show_effective_rank=show_effective_rank),
        figure_title        = 'Singular Value Evolution Across Training Epochs',
        legend_title        = legend_title,
        figsize_per_subplot = figsize_per_subplot,
    ), save_path=save_path, dpi=dpi)

def visualize_effective_ranks(
    evolutions:          Dict[str, Dict],
    figsize_per_subplot: Tuple[float, float] = (3.8, 2.6),
    save_path:           Optional[str]       = None,
    dpi:                 int                 = 150,
) -> plt.Figure:
    return visualize_evolution_grid(evolutions, PlotSpec(
        fn                  = effective_rank_plot_fn,
        col_multiplier      = 1,
        figure_title        = 'Effective Rank Evolution Across Training Epochs',
        legend_title        = None,
        figsize_per_subplot = figsize_per_subplot,
    ), save_path=save_path, dpi=dpi)

def visualize_condition_numbers(
    evolutions:          Dict[str, Dict],
    figsize_per_subplot: Tuple[float, float] = (3.8, 2.6),
    log_scale:           bool                = True,
    save_path:           Optional[str]       = None,
    dpi:                 int                 = 150,
) -> plt.Figure:
    return visualize_evolution_grid(evolutions, PlotSpec(
        fn                  = condition_number_plot_fn,
        kwargs              = dict(log_scale=log_scale),
        col_multiplier      = 1,
        figure_title        = 'Condition Number Evolution Across Training Epochs',
        legend_title        = None,
        figsize_per_subplot = figsize_per_subplot,
    ), save_path=save_path, dpi=dpi)

def visualize_angular_evolution(
    evolutions:          Dict[str, Dict],
    vector_type:         str                 = 'both',
    top_k_to_show:       int                 = 5,
    figsize_per_subplot: Tuple[float, float] = (3.8, 2.6),
    legend_title:        Optional[str]       = 'Singular vectors',
    save_path:           Optional[str]       = None,
    dpi:                 int                 = 150,
) -> plt.Figure:
    # Note: angular_data argument is gone — data lives in evolutions now
    subtitles = {
        'left' : 'Left Singular Vectors (U)',
        'right': 'Right Singular Vectors (V)',
        'both' : 'Left (U) and Right (V) Singular Vectors',
    }
    return visualize_evolution_grid(evolutions, PlotSpec(
        fn                  = individual_angles_plot_fn,
        kwargs              = dict(vector_type=vector_type,
                                   top_k_to_show=top_k_to_show),
        col_multiplier      = 2 if vector_type == 'both' else 1,
        figure_title        = (f'Individual Singular Vector Angular Change Across Training Epochs\n'
                               f'{subtitles[vector_type]}'),
        legend_title        = legend_title,
        figsize_per_subplot = figsize_per_subplot,
    ), save_path=save_path, dpi=dpi)

def visualize_principal_angles(
    evolutions:          Dict[str, Dict],
    vector_type:         str                 = 'both',
    top_k_to_show:       int                 = 5,
    figsize_per_subplot: Tuple[float, float] = (3.8, 2.6),
    legend_title:        Optional[str]       = 'Principal angles',
    save_path:           Optional[str]       = None,
    dpi:                 int                 = 150,
) -> plt.Figure:
    subtitles = {
        'left' : 'Left Singular Vectors (U)',
        'right': 'Right Singular Vectors (V)',
        'both' : 'Left (U) and Right (V) Singular Vectors',
    }
    return visualize_evolution_grid(evolutions, PlotSpec(
        fn                  = principal_angles_plot_fn,
        kwargs              = dict(vector_type=vector_type,
                                   top_k_to_show=top_k_to_show),
        col_multiplier      = 2 if vector_type == 'both' else 1,
        figure_title        = (f'Principal Angles Between Consecutive Singular Vector Space Across Training Epochs\n'
                               f'{subtitles[vector_type]}'),
        legend_title        = legend_title,
        figsize_per_subplot = figsize_per_subplot,
    ), save_path=save_path, dpi=dpi)

def visualize_grassmann_distance(
    evolutions:          Dict[str, Dict],
    vector_type:         str                 = 'both',
    figsize_per_subplot: Tuple[float, float] = (3.8, 2.6),
    save_path:           Optional[str]       = None,
    dpi:                 int                 = 150,
) -> plt.Figure:
    subtitles = {
        'left' : 'Left Singular Vectors (U)',
        'right': 'Right Singular Vectors (V)',
        'both' : 'Left (U) and Right (V) Singular Vectors',
    }
    return visualize_evolution_grid(evolutions, PlotSpec(
        fn                  = grassmann_dist_plot_fn,
        kwargs              = dict(vector_type=vector_type),
        col_multiplier      = 1,  # both U and V on the same axes
        figure_title        = (f'Grassmann Geodesic Distance Between Consecutive Singular Vector Space Across Training Epochs\n'
                               f'{subtitles[vector_type]}'),
        legend_title        = None,
        figsize_per_subplot = figsize_per_subplot,
    ), save_path=save_path, dpi=dpi)

def visualize_spectrum_at_epoch(
    evolutions: Dict[str, Dict],
    figsize_per_subplot: Tuple[float, float] = (3.8, 2.6),
    epoch_idx: int = -1,
    log_scale: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    return visualize_evolution_grid(evolutions, PlotSpec(
        fn           = sv_spectrum_plot_fn,
        kwargs       = dict(epoch_idx=epoch_idx, log_scale=log_scale),
        figure_title = f'Singular Value Spectrum (Epoch {evolutions[next(iter(evolutions))]["epochs"][epoch_idx]})',
        legend_title = None,
        figsize_per_subplot = figsize_per_subplot,
    ),save_path=save_path, dpi=dpi)

# ─────────────────────────────────────────────────────────────────────────── #
# 5. Quick-plot helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _name(evolution: Dict) -> str:
    """Derive a short display title from the stored weight name."""
    return evolution.get('name', 'weight').split('.')[-2]

def plot_sv(
    evolution:    Dict,
    top_k:        int   = 10,
    show_eff_rank: bool = True,
    legend_on:   bool = True,
    consistent_coloring: bool = True,
    figsize:      Tuple[float, float] = (16, 6),
) -> plt.Figure:
    """Singular value evolution + optional effective-rank overlay."""
    fig, ax = plt.subplots(figsize=figsize)
    sv_plot_fn(
        axes_slice          = [ax],
        evolution           = evolution,
        title               = "Singular value evolution (" + _name(evolution) + ")",
        show_xlabel         = True,
        show_ylabel         = True,
        top_k_to_show       = top_k,
        show_effective_rank = show_eff_rank,
        consisistent_coloring = consistent_coloring,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles and legend_on:
        ax.legend(fontsize=6.5, loc='upper right', framealpha=0.85)
    fig.tight_layout()
    plt.show()


def plot_effective_rank(
    evolution: Dict,
    figsize:   Tuple[float, float] = (16, 6),
) -> plt.Figure:
    """Effective rank over epochs."""
    fig, ax = plt.subplots(figsize=figsize)
    effective_rank_plot_fn(
        axes_slice  = [ax],
        evolution   = evolution,
        title       = "Effective rank (" + _name(evolution) + ")",
        show_xlabel = True,
        show_ylabel = True,
    )
    fig.tight_layout()
    plt.show()

def plot_condition_number(
    evolution: Dict,
    figsize:   Tuple[float, float] = (16, 6),
) -> plt.Figure:
    """Condition number over epochs (auto log-scale)."""
    fig, ax = plt.subplots(figsize=figsize)
    condition_number_plot_fn(
        axes_slice  = [ax],
        evolution   = evolution,
        title       = "Condition number (" + _name(evolution) + ")",
        show_xlabel = True,
        show_ylabel = True,
    )
    fig.tight_layout()
    plt.show()

def plot_individual_angles(
    evolution:   Dict,
    vector_type: str = 'both',      # 'left' | 'right' | 'both'
    top_k:       int = 5,
    figsize:     Tuple[float, float] = (16, 6),
    legend_on:   bool = True,
    consistent_coloring: bool = True,
) -> plt.Figure:
    """
    Individual singular vector angular changes between consecutive epochs.
    vector_type='both' produces a side-by-side (1 x 2) figure.
    """
    ncols = 2 if vector_type == 'both' else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    axes_slice = list(np.atleast_1d(axes))   # always a list

    individual_angles_plot_fn(
        axes_slice  = axes_slice,
        evolution   = evolution,
        title       = "Individual singular vector angular changes (" + _name(evolution) + ")",
        show_xlabel = True,
        show_ylabel = True,
        vector_type = vector_type,
        top_k_to_show = top_k,
        consistent_coloring=consistent_coloring,
    )

    handles, labels = axes_slice[0].get_legend_handles_labels()
    if legend_on == True and handles:
        axes_slice[0].legend(fontsize=6.5, loc='upper right', framealpha=0.85)

    fig.tight_layout()
    plt.show()


def plot_principal_angles(
    evolution:   Dict,
    vector_type: str = 'both',      # 'left' | 'right' | 'both'
    top_k:       int = 5,
    figsize:     Tuple[float, float] = (16, 6),
    legend_on:   bool = True,
    consistent_coloring: bool = True,
) -> plt.Figure:
    """
    Principal angles between consecutive epoch subspaces.
    vector_type='both' produces a side-by-side (1 x 2) figure.
    """
    ncols = 2 if vector_type == 'both' else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    axes_slice = list(np.atleast_1d(axes))

    principal_angles_plot_fn(
        axes_slice    = axes_slice,
        evolution     = evolution,
        title         = "Principal angles (" + _name(evolution) + ")",
        show_xlabel   = True,
        show_ylabel   = True,
        vector_type   = vector_type,
        top_k_to_show = top_k,
        consistent_coloring=consistent_coloring,
    )

    handles, labels = axes_slice[0].get_legend_handles_labels()
    if legend_on == True and handles:
        axes_slice[0].legend(fontsize=6.5, loc='upper right', framealpha=0.85)

    fig.tight_layout()
    plt.show()

def plot_grassmann(
    evolution:   Dict,
    vector_type: str  = 'both',     # 'left' | 'right' | 'both'
    figsize:     Tuple[float, float] = (16, 6),
) -> plt.Figure:
    """Grassmann geodesic distance between consecutive subspaces."""
    fig, ax = plt.subplots(figsize=figsize)
    grassmann_dist_plot_fn(
        axes_slice  = [ax],
        evolution   = evolution,
        title       = "Grassmann geodesic distance (" + _name(evolution) + ")",
        show_xlabel = True,
        show_ylabel = True,
        vector_type = vector_type,
    )
    fig.tight_layout()
    plt.show()