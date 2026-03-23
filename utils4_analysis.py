# ════════════════════════════════════════════════════════════════════════════════
# SVD Analysis
# ════════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Union, Literal, Tuple, Optional, Any
from dataclasses import dataclass

from utils2_svd import compute_svd, compute_effective_rank
from utils3_visual import parse_layer_info, organize_weights_for_plotting
from utils_Factored_Conv_Tas import FactoredConv1d

def visualize_singular_values_subplots(
    linear_weights:      Dict[str, torch.Tensor],
    log_scale:           bool                  = False,
    figsize_per_subplot: Tuple[float, float]   = (3.5, 2.8),
    figure_title:        str                   = 'Singular Value Spectrum',
    save_path:           Optional[str]        = None,
    dpi:                 int                   = 150,
) -> plt.Figure:
    """
    Singular value scree plot for every weight in a single checkpoint.

    Reuses the same grid geometry as visualize_evolution_grid — conv weights
    are grouped by layer index and type; non-conv weights fill leading rows.
    No evolutions dict or epoch data required.

    Args:
        linear_weights:      {weight_name: Tensor} from one state_dict.
        log_scale:           Apply log scale to all y-axes.
        figsize_per_subplot: (width, height) per cell in inches.
        figure_title:        Figure suptitle.
        dpi:                 Resolution for display / optional saving.

    Returns:
        plt.Figure
    """

    # ── 1. Grid geometry (identical arithmetic to visualize_evolution_grid) ──
    (non_conv_weights,
     conv_weights_by_idx,
     layer_type_order,
     base_num_cols,
     rows_per_index) = organize_weights_for_plotting(linear_weights)

    num_layer_types      = len(layer_type_order)
    sorted_layer_indices = sorted(conv_weights_by_idx.keys())

    non_conv_rows = (
        len(non_conv_weights) // base_num_cols + 1
        if non_conv_weights else 0
    )

    if num_layer_types == 2:
        conv_rows = (len(sorted_layer_indices) + 1) // 2
    else:
        conv_rows = len(sorted_layer_indices) * rows_per_index

    total_rows = non_conv_rows + conv_rows
    num_cols   = base_num_cols

    # ── 2. Figure ────────────────────────────────────────────────────────────
    w, h = figsize_per_subplot
    fig, axes = plt.subplots(
        total_rows, num_cols,
        figsize = (num_cols * w, total_rows * h),
        squeeze = False,
    )

    # ── 3. Per-slot drawing helper ───────────────────────────────────────────
    def _draw(row: int, col: int, weight_name: str, title: str,
              show_xlabel: bool, show_ylabel: bool) -> None:
        ax     = axes[row, col]
        weight = linear_weights[weight_name]

        # SVD
        svd_result  = compute_svd(weight)
        sv          = svd_result['S'].cpu().numpy()
        eff_rank    = compute_effective_rank(svd_result['S']).cpu().item()

        # Scree plot
        indices = np.arange(1, len(sv) + 1)
        ax.plot(indices, sv, 'o-', color='steelblue', linewidth=1.8, markersize=3.5)
        ax.axvline(eff_rank, color='red', linestyle='--', linewidth=1.2,
                   label=f'Eff. Rank: {eff_rank:.2f}')

        # Stats annotation
        cond = sv[0] / sv[-1] if sv[-1] > 0 else float('inf')
        ax.text(
            0.98, 0.97,
            f'Max:  {sv[0]:.2e}\nMin:  {sv[-1]:.2e}\nCond: {cond:.2e}',
            transform           = ax.transAxes,
            verticalalignment   = 'top',
            horizontalalignment = 'right',
            bbox                = dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize            = 7,
        )

        # Formatting
        shape_str = 'x'.join(str(d) for d in weight.shape)
        ax.set_title(f'{title}\n({shape_str})', fontsize=9, fontweight='bold')
        ax.set_xlim(1, len(sv))
        ax.set_ylim(bottom=None if log_scale else 0)
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

    def _hide(row: int, col: int) -> None:
        axes[row, col].axis('off')

    current_row = 0

    # ── 4. Non-conv weights ──────────────────────────────────────────────────
    if non_conv_weights:
        for flat_idx, weight_name in enumerate(sorted(non_conv_weights)):
            row_off  = flat_idx // base_num_cols
            col      = flat_idx  % base_num_cols
            info     = parse_layer_info(weight_name)
            title    = (info['layer_type'].capitalize()
                        if info['layer_type'] else weight_name.split('.')[-2])
            _draw(
                row         = current_row + row_off,
                col         = col,
                weight_name = weight_name,
                title       = title,
                show_xlabel = (current_row + row_off == total_rows - 1),
                show_ylabel = (col == 0),
            )

        for empty in range(len(non_conv_weights), non_conv_rows * base_num_cols):
            _hide(current_row + empty // base_num_cols, empty % base_num_cols)

        current_row += non_conv_rows

    # ── 5a. Conv weights — 2 type layout (2 indices side-by-side per row) ───
    if num_layer_types == 2:
        for i in range(0, len(sorted_layer_indices), 2):
            for j in range(2):
                if i + j >= len(sorted_layer_indices):
                    for t in range(len(layer_type_order)):
                        _hide(current_row, j * len(layer_type_order) + t)
                    continue

                layer_idx     = sorted_layer_indices[i + j]
                layer_weights = conv_weights_by_idx[layer_idx]

                for type_idx, layer_type in enumerate(layer_type_order):
                    col = j * len(layer_type_order) + type_idx

                    if layer_type in layer_weights:
                        _draw(
                            row         = current_row,
                            col         = col,
                            weight_name = layer_weights[layer_type],
                            title       = f'Layer {layer_idx} - {layer_type.capitalize()}',
                            show_xlabel = (current_row == total_rows - 1),
                            show_ylabel = (col == 0),
                        )
                    else:
                        _hide(current_row, col)

            current_row += 1

    # ── 5b. Conv weights — general layout ────────────────────────────────────
    else:
        for layer_idx in sorted_layer_indices:
            layer_weights = conv_weights_by_idx[layer_idx]
            type_counter  = 0

            for row_off in range(rows_per_index):
                for col in range(base_num_cols):
                    if type_counter >= len(layer_type_order):
                        _hide(current_row + row_off, col)
                        continue

                    layer_type   = layer_type_order[type_counter]
                    type_counter += 1

                    if layer_type in layer_weights:
                        _draw(
                            row         = current_row + row_off,
                            col         = col,
                            weight_name = layer_weights[layer_type],
                            title       = f'Layer {layer_idx} - {layer_type.capitalize()}',
                            show_xlabel = (current_row + row_off == total_rows - 1),
                            show_ylabel = (col == 0),
                        )
                    else:
                        _hide(current_row + row_off, col)

            current_row += rows_per_index

    # ── 6. Suptitle + layout ─────────────────────────────────────────────────
    fig.suptitle(figure_title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

@dataclass
class FactoredWeight:
    """
    Stores the rank-k SVD factorization of one weight matrix.

    A @ B  reconstructs the best rank-k approximation of the original W.

    Attributes:
        A          : Left  factor  U_k @ diag(sigma_k^0.5)   shape [n, k]
        B          : Right factor  diag(sigma_k^0.5) @ V_k^T shape [k, m]
        rank_k     : Effective rank (rounded integer)
        orig_shape : Original weight shape (n, m)
        compression: Parameter compression ratio  (n*m) / (k*(n+m))
    """
    U_k:           torch.Tensor
    S_k:           torch.Tensor
    Vt_k:          torch.Tensor


def factorize_linear_weights(
    linear_weights: Dict[str, torch.Tensor],
    min_rank:       int  = 1,
    max_rank_frac:  float = 1.0,    # cap rank at this fraction of min(n,m)
) -> Dict[str, FactoredWeight]:
    """
    For each weight W in linear_weights:
      1. Compute full SVD.
      2. Round effective rank to nearest integer k  (clamped to [min_rank, max_rank]).
      3. Return a dict of FactoredWeight objects.

    Args:
        linear_weights: {name: Tensor (n, m)} from a single state_dict.
        min_rank:       Hard floor on k (avoids degenerate rank-1 tensors).
        max_rank_frac:  Soft ceiling — k <= max_rank_frac * min(n, m).
                        Set to 1.0 to allow full rank (no-op for full-rank matrices).

    Returns:
        {name: FactoredWeight}
    """
    factored: Dict[str, FactoredWeight] = {}

    Orig_params_dim = 0
    Fact_params_dim = 0
    for name, weight in linear_weights.items():
        W  = weight.float()                    # ensure float32 for SVD stability
        n, m = W.shape

        # ── SVD ────────────────────────────────────────────────────────── #
        svd_result = compute_svd(W)            # your existing function
        U  = svd_result['U']                   # [n, r]
        S  = svd_result['S']                   # [r]
        Vt = svd_result['Vt']                  # [r, m]

        # ── Effective rank → integer k ──────────────────────────────────── #
        eff_rank = compute_effective_rank(S)
        k = int(round(eff_rank.item()))
        k = max(min_rank, k)
        k = min(k, int(max_rank_frac * min(n, m)))
        k = min(k, S.shape[0])                 # can't exceed stored rank

        # ── Truncate ────────────────────────────────────────────────────── #
        U_k  = U[:, :k]                        # [n, k]
        S_k  = S[:k]                           # [k]
        Vt_k = Vt[:k, :]                       # [k, m]

        # ── Compression ratio ────────────────────────────────────────────── #
        orig_params = n * m
        fact_params = k * (n + m)

        factored[name] = FactoredWeight(
            U_k         = U_k.to(weight.dtype),
            S_k         = S_k.to(weight.dtype),
            Vt_k        = Vt_k.to(weight.dtype)
        )

        print(
            f'{name:<55}  '
            f'shape ({n:4d},{m:4d})  '
            f'k={k:4d}  '
        )

        Orig_params_dim += orig_params
        Fact_params_dim += fact_params

    return factored, Orig_params_dim, Fact_params_dim

def build_factored_state_dict(
    original_state_dict: Dict[str, torch.Tensor],
    factored_weights:    Dict[str, FactoredWeight],
    factor_type:         str = 'svd'
) -> Dict[str, torch.Tensor]:
    """
    Construct a new state_dict suitable for a factored Conv-TasNet variant.

    For each factored weight 'encoder.layers.0.linear.weight':
      - Remove the original key
      - Insert  'encoder.layers.0.linear.weight_U'  (shape [n, k])
                'encoder.layers.0.linear.weight_S'  (shape [k])
                'encoder.layers.0.linear.weight_Vt'  (shape [k, m])

    All non-linear keys (biases, norms, etc.) are carried over unchanged.

    Returns:
        New state_dict with A/B pairs substituted for every factored weight.
    """
    new_sd = {}

    for key, tensor in original_state_dict.items():
        if key in factored_weights:
            fw = factored_weights[key]
            module_path = key.rsplit('.', 1)[0]
            if factor_type == 'svd':
                new_sd[f'{module_path}.A'] = fw.U_k * fw.S_k.unsqueeze(0)   # [n, k]
                new_sd[f'{module_path}.B'] = fw.Vt_k    # [k, m]
            elif factor_type == 'sqrt_svd':
                sqrt_S_k = torch.sqrt(fw.S_k)
                new_sd[f'{module_path}.A'] = fw.U_k  * sqrt_S_k.unsqueeze(0)  # [n, k]
                new_sd[f'{module_path}.B'] = sqrt_S_k.unsqueeze(1) * fw.Vt_k  # [k, m]
        else:
            new_sd[key] = tensor.clone()   # bias, LayerNorm, etc. unchanged

    return new_sd

def load_state_dict_with_resize(
    model:      nn.Module,
    state_dict: Dict[str, torch.Tensor],
    verbose:    bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load a state_dict into a model, allowing size mismatches.

    For every key in state_dict:
      - If the shape matches  → standard copy (same as load_state_dict)
      - If the shape differs  → reallocate the parameter/buffer to the new shape
      - If the key is missing in the model → reported and skipped

    For every key in the model not present in state_dict → reported and skipped (original value is preserved).

    Parameters
    ----------
    model      : target nn.Module
    state_dict : source tensors (keys must match model's named_parameters /
                 named_buffers exactly, i.e. same dotted paths)
    verbose    : print a summary table of what was done

    Returns
    -------
    matched   : keys copied with matching shape
    resized   : keys copied with shape change
    skipped   : keys present in one side but not the other
    """
    model_params  = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    model_all     = {**model_params, **model_buffers}

    matched: List[str] = []
    resized: List[str] = []
    skipped: List[str] = []

    # ── Keys in state_dict but not in model ──────────────────────────────── #
    for key in state_dict:
        if key not in model_all:
            skipped.append(key)
            if verbose:
                print(f'  [SKIP  ] {key} — not found in model')

    # ── Keys in model but not in state_dict ──────────────────────────────── #
    for key in model_all:
        if key not in state_dict:
            skipped.append(key)
            if verbose:
                print(f'  [SKIP  ] {key} — not found in state_dict (kept original)')

    # ── Load every key that exists on both sides ──────────────────────────── #
    for key, src_tensor in state_dict.items():
        if key not in model_all:
            continue   # already reported above

        src = src_tensor.detach()

        is_param  = key in model_params
        dst       = model_all[key]

        if dst.shape == src.shape:
            # ── Matching shape: simple in-place copy ─────────────────────── #
            with torch.no_grad():
                dst.copy_(src)
            matched.append(key)
            if verbose:
                print(f'  [OK    ] {key:<55} {tuple(src.shape)}')

        else:
            # ── Mismatched shape: reallocate ──────────────────────────────── #
            # We cannot copy_ into a different shape.
            # We must navigate to the parent module and call setattr()
            # so that the model holds the new tensor as its parameter/buffer.
            _replace_tensor(model, key, src, is_param=is_param)
            resized.append(key)
            if verbose:
                print(
                    f'  [RESIZE] {key:<55} '
                    f'{tuple(dst.shape)} → {tuple(src.shape)}'
                )

    # ── Summary ──────────────────────────────────────────────────────────── #
    if verbose:
        print(
            f'\nDone.  matched={len(matched)}  '
            f'resized={len(resized)}  skipped={len(skipped)}'
        )

    return matched, resized, skipped


def _replace_tensor(
    model:    nn.Module,
    key:      str,
    tensor:   torch.Tensor,
    is_param: bool,
) -> None:
    """
    Navigate to the parent module of `key` and rebind the attribute.

    Given key = 'encoder.layers.0.linear.weight':
        parent  = model.encoder.layers[0].linear
        attr    = 'weight'

    For nn.Parameter:  wraps tensor in nn.Parameter (preserves requires_grad)
    For buffer:        registers as plain tensor (no grad)
    """
    # Split dotted path into parent path and attribute name
    if '.' in key:
        parent_path, attr_name = key.rsplit('.', 1)
        parent = _get_module(model, parent_path)
    else:
        parent    = model
        attr_name = key

    new_tensor = tensor.to(dtype=getattr(parent, attr_name).dtype)

    if is_param:
        # Must use nn.Parameter to keep it in _parameters and requires_grad
        requires_grad = getattr(parent, attr_name).requires_grad
        setattr(parent, attr_name, nn.Parameter(new_tensor, requires_grad=requires_grad))
    else:
        # Buffers: use register_buffer to keep them in _buffers
        parent.register_buffer(attr_name, new_tensor)


def _get_module(model: nn.Module, dotted_path: str) -> nn.Module:
    """Traverse model along a dotted attribute path."""
    obj = model
    for part in dotted_path.split('.'):
        obj = getattr(obj, part)
    return obj

def sync_factored_ranks_full(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """
    Walk ALL FactoredConv1d layers and sync .rank from A.shape[1] regardless of whether a resize happened.
    """
    changed = {}
    for name, module in model.named_modules():
        if isinstance(module, FactoredConv1d):
            live_rank = module.A.shape[1]
            if module.rank != live_rank:
                changed[name] = (module.rank, live_rank)
                module.rank   = live_rank
    return changed