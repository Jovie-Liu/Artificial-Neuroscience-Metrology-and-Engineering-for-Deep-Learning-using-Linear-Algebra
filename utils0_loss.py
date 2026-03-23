# ════════════════════════════════════════════════════════════════════════════════
# Plot Training Loss
# ════════════════════════════════════════════════════════════════════════════════


import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Literal, Tuple, Optional, Any
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import copy

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def load_tensorboard_scalars(
    log_dir: str,
    tags: Tuple[str, ...] = ('Epoch/Train_Loss', 'Epoch/Val_Loss')
) -> dict:
    """
    Load scalar data from TensorBoard event files.

    Args:
        log_dir: Path to the TensorBoard log directory
        tags:    Tuple of scalar tag names to extract

    Returns:
        Dictionary mapping tag -> {'epochs': list, 'values': list}
        Epochs are converted to 1-indexed for human-readable x-axis.

    Notes:
        - In the training script, scalars are logged with 0-indexed epoch as
          the step: writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
          where epoch starts at 0. We convert to 1-indexed here.
    """
    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    available_tags = ea.Tags().get('scalars', [])
    print(f"Available TensorBoard scalar tags: {available_tags}")

    results = {}

    for tag in tags:
        if tag not in available_tags:
            print(f"  Warning: Tag '{tag}' not found in TensorBoard logs.")
            results[tag] = {'epochs': [], 'values': []}
            continue

        events = ea.Scalars(tag)

        # step is 0-indexed epoch → convert to 1-indexed
        epochs = [e.step + 1 for e in events]
        values = [e.value   for e in events]

        results[tag] = {'epochs': epochs, 'values': values}
        print(f"  Loaded '{tag}': {len(epochs)} data points "
              f"(epoch {min(epochs)} → {max(epochs)})")

    return results

def load_best_val_loss_from_checkpoints(
    checkpoint_dir: str
) -> dict:
    """
    Scan all epoch checkpoints and extract the running best_val_loss recorded at each epoch.

    Args:
        checkpoint_dir: Directory containing checkpoint_epoch_N.pth files

    Returns:
        Dictionary {'epochs': list[int], 'values': list[float]}
        Epochs are 1-indexed (matching the filename convention).

    Notes:
        - The training script saves checkpoint_epoch_{epoch+1}.pth where epoch is 0-indexed, so the filename number is already 1-indexed.
        - best_val_loss stored in checkpoint is the running minimum val loss seen up to (and including) that epoch.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pth'),
        key=lambda p: int(p.stem.split('_')[-1])   # sort by epoch number
    )

    if not checkpoint_files:
        print(f"  Warning: No epoch checkpoints found in '{checkpoint_dir}'")
        return {'epochs': [], 'values': []}

    epochs = []
    best_val_losses = []

    for ckpt_path in checkpoint_files:
        # Extract the 1-indexed epoch number from filename
        epoch_num = int(ckpt_path.stem.split('_')[-1])

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        best_val_loss = checkpoint.get('best_val_loss', None)

        if best_val_loss is not None and best_val_loss != float('inf'):
            epochs.append(epoch_num)
            best_val_losses.append(float(best_val_loss))

    print(f"  Loaded best_val_loss from checkpoints: {len(epochs)} data points "
          f"(epoch {min(epochs) if epochs else 'N/A'} → "
          f"{max(epochs) if epochs else 'N/A'})")

    return {'epochs': epochs, 'values': best_val_losses}

# =============================================================================
# SECTION 2: PLOTTING
# =============================================================================

def plot_training_curves(
    log_dir: str,
    checkpoint_dir: str,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    dpi: int = 150,
    smooth_window: int = 1,
) -> plt.Figure:
    """
    Plot Train Loss, Val Loss (from TensorBoard) and best_val_loss (from checkpoints) all against epoch on a single figure.

    Args:
        log_dir:        TensorBoard log directory
        checkpoint_dir: Directory with checkpoint_epoch_N.pth files
        save_path:      If provided, save figure to this path
        figsize:        Figure size (width, height) in inches
        dpi:            Resolution for saved figure
        smooth_window:  Rolling average window for Train/Val loss curves
                        (1 = no smoothing)

    Returns:
        Matplotlib Figure object
    """
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading TensorBoard scalars...")
    tb_data = load_tensorboard_scalars(
        log_dir,
        tags=('Epoch/Train_Loss', 'Epoch/Val_Loss')
    )

    print("\nLoading best_val_loss from checkpoints...")
    ckpt_data = load_best_val_loss_from_checkpoints(checkpoint_dir)

    # ------------------------------------------------------------------
    # Smooth helper
    # ------------------------------------------------------------------
    def smooth(values: list, window: int) -> np.ndarray:
        """Apply centered rolling average."""
        if window <= 1 or len(values) < window:
            return np.array(values)
        kernel = np.ones(window) / window
        padded = np.pad(values, window // 2, mode='edge')
        return np.convolve(padded, kernel, mode='valid')[:len(values)]

    # ------------------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------------------
    train_epochs = np.array(tb_data['Epoch/Train_Loss']['epochs'])
    train_values = np.array(tb_data['Epoch/Train_Loss']['values'])
    train_smooth = smooth(train_values, smooth_window)

    val_epochs   = np.array(tb_data['Epoch/Val_Loss']['epochs'])
    val_values   = np.array(tb_data['Epoch/Val_Loss']['values'])
    val_smooth   = smooth(val_values, smooth_window)

    ckpt_epochs  = np.array(ckpt_data['epochs'])
    ckpt_values  = np.array(ckpt_data['values'])

    # ------------------------------------------------------------------
    # Figure layout: main loss axes + optional LR twin axis placeholder
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # ── Color palette ──────────────────────────────────────────────────
    COLOR_TRAIN      = '#4C8BF5'   # blue
    COLOR_VAL        = '#F5A623'   # orange
    COLOR_BEST_VAL   = '#E84040'   # red
    COLOR_GRID       = '#E0E0E0'

    # ── Train Loss ─────────────────────────────────────────────────────
    if len(train_epochs) > 0:
        # Raw (faint)
        if smooth_window > 1:
            ax.plot(
                train_epochs, train_values,
                color=COLOR_TRAIN, alpha=0.2, linewidth=0.8,
                label='_nolegend_'
            )
        # Smoothed (or raw if no smoothing)
        ax.plot(
            train_epochs, train_smooth,
            color=COLOR_TRAIN, linewidth=2.0,
            label='Train Loss' + (f' (smooth={smooth_window})' if smooth_window > 1 else '')
        )

    # ── Val Loss ───────────────────────────────────────────────────────
    if len(val_epochs) > 0:
        if smooth_window > 1:
            ax.plot(
                val_epochs, val_values,
                color=COLOR_VAL, alpha=0.2, linewidth=0.8,
                label='_nolegend_'
            )
        ax.plot(
            val_epochs, val_smooth,
            color=COLOR_VAL, linewidth=2.0,
            label='Val Loss' + (f' (smooth={smooth_window})' if smooth_window > 1 else '')
        )

    # ── Best Val Loss (from checkpoints) ───────────────────────────────
    if len(ckpt_epochs) > 0:
        ax.step(
            ckpt_epochs, ckpt_values,
            where='post',                       # staircase: best drops only downward
            color=COLOR_BEST_VAL, linewidth=2.0,
            linestyle='--',
            label='Best Val Loss (checkpoint)'
        )

        # Mark the global best epoch with a star
        best_idx   = np.argmin(ckpt_values)
        best_epoch = ckpt_epochs[best_idx]
        best_loss  = ckpt_values[best_idx]

        ax.scatter(
            best_epoch, best_loss,
            marker='*', s=220,
            color=COLOR_BEST_VAL, zorder=5,
            label=f'Global Best  (epoch {best_epoch}, loss {best_loss:.4f})'
        )

        # Annotate the best point
        ax.annotate(
            f' ★ {best_loss:.4f}\n   epoch {best_epoch}',
            xy=(best_epoch, best_loss),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=8,
            color=COLOR_BEST_VAL,
            arrowprops=dict(arrowstyle='->', color=COLOR_BEST_VAL, lw=1.2)
        )

    # ── Formatting ─────────────────────────────────────────────────────
    all_epochs = np.concatenate([
        e for e in [train_epochs, val_epochs, ckpt_epochs] if len(e) > 0
    ])
    if len(all_epochs) > 0:
        ax.set_xlim(left=max(0, all_epochs.min() - 0.5),
                    right=all_epochs.max() + 0.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss  (SI-SNR Loss)', fontsize=12)
    ax.set_title('Conv-TasNet · Training Curves', fontsize=14, fontweight='bold')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, linestyle='--', alpha=0.9)
    ax.set_axisbelow(True)

    legend = ax.legend(
        loc='upper right',
        fontsize=9,
        framealpha=0.92,
        edgecolor='#CCCCCC'
    )

    # ── Footer note ────────────────────────────────────────────────────
    fig.text(
        0.5, -0.02,
        'Train/Val Loss sourced from TensorBoard (Epoch/Train_Loss, Epoch/Val_Loss)  |  '
        'Best Val Loss sourced from checkpoint_epoch_N.pth files',
        ha='center', fontsize=7, color='#888888', style='italic'
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\nFigure saved to: {save_path}")

    return fig

# ════════════════════════════════════════════════════════════════════════════════
# Plot Training Loss — Round 1 + Round 2 (Fine-tune from best_epoch)
# ════════════════════════════════════════════════════════════════════════════════

def _smooth(values: list, window: int) -> np.ndarray:
    """Centered rolling average. window=1 → no-op."""
    if window <= 1 or len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]

def _plot_round(
    ax: plt.Axes,
    tb_data:      dict,
    ckpt_data:    dict,
    epoch_offset: int,
    smooth_window: int,
    colors:       dict,
    label_suffix: str,
    annotate_best: bool = True,
    best_annotation_side: str = 'right',  # 'right' or 'left'
) -> int:
    """
    Draw one round of training curves onto *ax*.

    Args:
        epoch_offset:  Added to every local epoch number (0 for round 1,
                       best_epoch_r1 for round 2).
        colors:        Dict with keys 'train', 'val', 'best_val'.
        label_suffix:  String appended to legend labels, e.g. ' [R1]'.
        annotate_best: Whether to draw the star + annotation for this round.
        best_annotation_side: Offset direction for the annotation text.

    Returns:
        global_best_epoch (int) for the caller to use as the next offset.
    """
    train_epochs_local = np.array(tb_data['Epoch/Train_Loss']['epochs'])
    train_values       = np.array(tb_data['Epoch/Train_Loss']['values'])

    val_epochs_local   = np.array(tb_data['Epoch/Val_Loss']['epochs'])
    val_values         = np.array(tb_data['Epoch/Val_Loss']['values'])

    ckpt_epochs_local  = np.array(ckpt_data['epochs'])
    ckpt_values        = np.array(ckpt_data['values'])

    # ── Apply global offset ────────────────────────────────────────────
    train_epochs = train_epochs_local + epoch_offset
    val_epochs   = val_epochs_local   + epoch_offset
    ckpt_epochs  = ckpt_epochs_local  + epoch_offset

    train_smooth = _smooth(train_values, smooth_window)
    val_smooth   = _smooth(val_values,   smooth_window)

    sw_tag = f' (smooth={smooth_window})' if smooth_window > 1 else ''

    # ── Train Loss ─────────────────────────────────────────────────────
    if len(train_epochs) > 0:
        if smooth_window > 1:
            ax.plot(train_epochs, train_values,
                    color=colors['train'], alpha=0.18, linewidth=0.8,
                    label='_nolegend_')
        ax.plot(train_epochs, train_smooth,
                color=colors['train'], linewidth=2.0,
                label=f'Train Loss{sw_tag}{label_suffix}')

    # ── Val Loss ───────────────────────────────────────────────────────
    if len(val_epochs) > 0:
        if smooth_window > 1:
            ax.plot(val_epochs, val_values,
                    color=colors['val'], alpha=0.18, linewidth=0.8,
                    label='_nolegend_')
        ax.plot(val_epochs, val_smooth,
                color=colors['val'], linewidth=2.0,
                label=f'Val Loss{sw_tag}{label_suffix}')

    # ── Best Val Loss staircase ────────────────────────────────────────
    best_epoch_global = None
    if len(ckpt_epochs) > 0:
        ax.step(ckpt_epochs, ckpt_values,
                where='post',
                color=colors['best_val'], linewidth=1.8,
                linestyle='--',
                label=f'Best Val Loss{label_suffix}')

        best_idx          = np.argmin(ckpt_values)
        best_epoch_global = int(ckpt_epochs[best_idx])
        best_loss         = float(ckpt_values[best_idx])

        if annotate_best:
            ax.scatter(best_epoch_global, best_loss,
                       marker='*', s=240,
                       color=colors['best_val'], zorder=6,
                       label=f'Global Best{label_suffix}  '
                             f'(ep {best_epoch_global}, {best_loss:.4f})')

            xoff = 10 if best_annotation_side == 'right' else -70
            ax.annotate(
                f' ★ {best_loss:.4f}\n   ep {best_epoch_global}',
                xy=(best_epoch_global, best_loss),
                xytext=(xoff, 10),
                textcoords='offset points',
                fontsize=8,
                color=colors['best_val'],
                arrowprops=dict(arrowstyle='->', color=colors['best_val'], lw=1.2)
            )

    return best_epoch_global


# =============================================================================
# SECTION 3: MAIN PLOTTING FUNCTION
# =============================================================================

def plot_training_curves_dual_round(
    # ── Round 1 ───────────────────────────────────────────────────────
    r1_log_dir:        str,
    r1_checkpoint_dir: str,
    # ── Round 2 ───────────────────────────────────────────────────────
    r2_log_dir:        str,
    r2_checkpoint_dir: str,
    # ── NEW: Continuity anchor values for Round 2 ─────────────────────
    r2_start_train_loss: Optional[float] = None,
    r2_start_val_loss:   Optional[float] = None,
    # ── Output ────────────────────────────────────────────────────────
    save_path:         Optional[str] = None,
    figsize:           Tuple[float, float] = (14, 5),
    dpi:               int = 150,
    smooth_window:     int = 1,
) -> plt.Figure:
    """
    Plot two rounds of training on a single figure with optional
    continuity anchor points at the Round 1 → Round 2 boundary.
 
    Args:
        r1_log_dir:           TensorBoard log dir for Round 1.
        r1_checkpoint_dir:    Checkpoint dir for Round 1.
        r2_log_dir:           TensorBoard log dir for Round 2.
        r2_checkpoint_dir:    Checkpoint dir for Round 2.
        r2_start_train_loss:  Optional float. When supplied, a synthetic
                              data point is prepended to Round 2's Train
                              Loss curve at the global best_epoch x position.
                              Set this to Round 1's train loss at best_epoch
                              so the curve is visually continuous.
        r2_start_val_loss:    Optional float. When supplied, a synthetic
                              data point is prepended to Round 2's Val Loss
                              curve AND to the Best Val Loss staircase at
                              global best_epoch. Set this to Round 1's
                              best_val_loss for visual continuity.
        save_path:            Optional file path to save the figure.
        figsize:              (width, height) in inches.
        dpi:                  Resolution for saved figure.
        smooth_window:        Rolling-average window (1 = off).
 
    Returns:
        Matplotlib Figure object.
    """
    # ── Color palettes ─────────────────────────────────────────────────
    COLORS_R1 = {
        'train':    '#4C8BF5',
        'val':      '#F5A623',
        'best_val': '#E84040',
    }
    COLORS_R2 = {
        'train':    '#1DB954',
        'val':      '#9B59B6',
        'best_val': '#E67E22',
    }
    COLOR_GRID      = '#E0E0E0'
    COLOR_SEPARATOR = '#555555'
 
    # ── Load data ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading Round 1 — TensorBoard scalars...")
    r1_tb   = load_tensorboard_scalars(r1_log_dir)
    print("\nLoading Round 1 — checkpoints...")
    r1_ckpt = load_best_val_loss_from_checkpoints(r1_checkpoint_dir)
 
    print("\nLoading Round 2 — TensorBoard scalars...")
    r2_tb   = load_tensorboard_scalars(r2_log_dir)
    print("\nLoading Round 2 — checkpoints...")
    r2_ckpt = load_best_val_loss_from_checkpoints(r2_checkpoint_dir)
    print("=" * 60)
 
    # ── Round 1 global best (= x-offset for Round 2) ──────────────────
    if len(r1_ckpt['epochs']) == 0:
        raise ValueError(
            "Round 1 checkpoint data is empty — cannot determine best_epoch offset."
        )
 
    r1_best_idx   = int(np.argmin(r1_ckpt['values']))
    r1_best_epoch = int(r1_ckpt['epochs'][r1_best_idx])
    r1_best_loss  = float(r1_ckpt['values'][r1_best_idx])
    r2_epoch_offset = r1_best_epoch
 
    print(f"\nRound 1 global best → epoch {r1_best_epoch}  (loss {r1_best_loss:.4f})")
    print(f"Round 2 epoch offset = {r2_epoch_offset}  "
          f"(R2 local ep 1 → global ep {r2_epoch_offset + 1})")
 
    # ── NEW: Inject anchor points into Round 2 data ────────────────────
    #
    #   We deep-copy R2 dicts so the originals are never mutated.
    #   Local epoch 0 + r2_epoch_offset = r1_best_epoch (global),
    #   which is exactly the separator line position.
    #
    r2_tb_plot   = copy.deepcopy(r2_tb)
    r2_ckpt_plot = copy.deepcopy(r2_ckpt)
 
    if r2_start_train_loss is not None:
        # Prepend epoch=0, value=r2_start_train_loss to Train Loss
        r2_tb_plot['Epoch/Train_Loss']['epochs'].insert(0, 0)
        r2_tb_plot['Epoch/Train_Loss']['values'].insert(0, r2_start_train_loss)
        print(f"\nR2 Train Loss anchor injected at local ep 0 "
              f"(global ep {r1_best_epoch}): {r2_start_train_loss:.4f}")
 
    if r2_start_val_loss is not None:
        # Prepend epoch=0, value=r2_start_val_loss to Val Loss
        r2_tb_plot['Epoch/Val_Loss']['epochs'].insert(0, 0)
        r2_tb_plot['Epoch/Val_Loss']['values'].insert(0, r2_start_val_loss)
        # Prepend epoch=0, value=r2_start_val_loss to Best Val Loss staircase
        r2_ckpt_plot['epochs'].insert(0, 0)
        r2_ckpt_plot['values'].insert(0, r2_start_val_loss)
        print(f"R2 Val Loss + Best Val Loss anchor injected at local ep 0 "
              f"(global ep {r1_best_epoch}): {r2_start_val_loss:.4f}")
 
    # ── Figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
 
    # ── Plot Round 1 ──────────────────────────────────────────────────
    _plot_round(
        ax=ax, tb_data=r1_tb, ckpt_data=r1_ckpt,
        epoch_offset=0, smooth_window=smooth_window,
        colors=COLORS_R1, label_suffix=' [R1]',
        annotate_best=True, best_annotation_side='right',
    )
 
    # ── Plot Round 2 (with anchor-injected copies) ─────────────────────
    _plot_round(
        ax=ax, tb_data=r2_tb_plot, ckpt_data=r2_ckpt_plot,  # ← patched copies
        epoch_offset=r2_epoch_offset, smooth_window=smooth_window,
        colors=COLORS_R2, label_suffix=' [R2]',
        annotate_best=True, best_annotation_side='right',
    )
 
    # ── Vertical separator ─────────────────────────────────────────────
    ax.axvline(x=r1_best_epoch, color=COLOR_SEPARATOR,
               linewidth=1.5, linestyle=':', zorder=4,
               label=f'Fine-tune start (ep {r1_best_epoch})')
    ax.text(
        r1_best_epoch + 0.15, 1.0, '← R1   R2 →',
        fontsize=8, color=COLOR_SEPARATOR, va='top', style='italic',
        transform=ax.get_xaxis_transform()
    )
 
    # ── X-axis limits ──────────────────────────────────────────────────
    r1_train_g = np.array(r1_tb['Epoch/Train_Loss']['epochs'])
    r1_val_g   = np.array(r1_tb['Epoch/Val_Loss']['epochs'])
    r2_train_g = np.array(r2_tb_plot['Epoch/Train_Loss']['epochs']) + r2_epoch_offset
    r2_val_g   = np.array(r2_tb_plot['Epoch/Val_Loss']['epochs'])   + r2_epoch_offset
    r1_ckpt_g  = np.array(r1_ckpt['epochs'])
    r2_ckpt_g  = np.array(r2_ckpt_plot['epochs']) + r2_epoch_offset
 
    all_global = np.concatenate([
        a for a in [r1_train_g, r1_val_g, r2_train_g,
                    r2_val_g, r1_ckpt_g, r2_ckpt_g] if len(a) > 0
    ])
    if len(all_global) > 0:
        ax.set_xlim(left  = max(0, all_global.min() - 0.5),
                    right = all_global.max() + 0.5)
 
    # ── Shaded phase backgrounds ───────────────────────────────────────
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0],       r1_best_epoch, alpha=0.04, color='#4C8BF5', zorder=0)
    ax.axvspan(r1_best_epoch, xlim[1],       alpha=0.04, color='#1DB954', zorder=0)
 
    # ── Axes formatting ────────────────────────────────────────────────
    ax.set_xlabel('Epoch (global)', fontsize=12)
    ax.set_ylabel('Loss  (SI-SNR Loss)', fontsize=12)
    ax.set_title('Conv-TasNet · Training Curves  [Round 1 + Round 2 Fine-tune]',
                 fontsize=14, fontweight='bold')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, linestyle='--', alpha=0.9)
    ax.set_axisbelow(True)
 
    # ── Legend ─────────────────────────────────────────────────────────
    r1_patch = mpatches.Patch(facecolor='#D6E4FF', edgecolor=COLOR_SEPARATOR,
                               label='Phase R1', linewidth=0.8)
    r2_patch = mpatches.Patch(facecolor='#D4EDDA', edgecolor=COLOR_SEPARATOR,
                               label='Phase R2', linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    handles += [r1_patch, r2_patch]
    labels  += ['Phase: Round 1', 'Phase: Round 2']
    ax.legend(handles, labels, loc='upper right', fontsize=8.5,
              framealpha=0.93, edgecolor='#CCCCCC', ncol=2)
 
    # ── Footer ─────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.02,
        'Round 1: TensorBoard + checkpoint_epoch_N.pth  |  '
        'Round 2: same sources, x-axis offset by R1 best_epoch  |  '
        'Anchor points manually supplied via r2_start_train_loss / r2_start_val_loss',
        ha='center', fontsize=7, color='#888888', style='italic'
    )
 
    plt.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\nFigure saved → {save_path}")
 
    return fig