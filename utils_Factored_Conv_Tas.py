import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import permutations
from typing import Optional, Tuple, Dict
from pathlib import Path
import time
import os
import soundfile as sf
from contextlib import contextmanager

class FactoredConv1d(nn.Module):
    """
    Drop-in replacement for nn.Conv1d(in_ch, out_ch, kernel_size=1).

    Normal operation
    ----------------
    Trains A [out_ch, rank] and B [rank, in_ch].
    forward() computes F.conv1d with W_eff = A @ B — autograd differentiates through A, B normally.

    Gradient-probe mode  (entered via .grad_probe_mode() context manager)
    ------------------
    A temporary LEAF parameter W_probe = (A @ B).detach() is created and substituted into forward(). 
    One forward+backward pass is run to deposit the exact ∂L/∂W into W_probe.grad. A and B are NEVER modified or
    re-initialised here. W_probe is discarded when the context exits.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        rank:         int,
        bias:         bool = True,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.rank         = rank

        # ── Factored parameters — always trained ──────────────────────────── #
        self.A = nn.Parameter(torch.empty(out_channels, rank))
        self.B = nn.Parameter(torch.empty(rank, in_channels))
        nn.init.kaiming_uniform_(self.A, a=0.01)
        nn.init.kaiming_uniform_(self.B, a=0.01)

        # ── Bias — independent of factorisation ───────────────────────────── #
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # ── Probe slot — None outside context manager ─────────────────────── #
        # When inside grad_probe_mode(), this holds the temporary leaf W.
        self._W_probe: Optional[nn.Parameter] = None

    # ── Forward ──────────────────────────────────────────────────────────── #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._W_probe is not None:
            # Probe mode: use the temporary leaf W; A,B not in this graph
            W_eff = self._W_probe                          # [out_ch, in_ch]
        else:
            # Normal mode: compose A @ B; autograd trains A, B
            W_eff = self.A @ self.B                        # [out_ch, in_ch]

        return F.conv1d(x, W_eff.unsqueeze(-1), self.bias)

    # ── Gradient-probe context manager ───────────────────────────────────── #

    @contextmanager
    def grad_probe_mode(self):
        """
        Context manager that temporarily installs a leaf parameter W_probe = detach(A @ B) into the forward pass.

        Inside the context:
        - forward() uses W_probe, NOT A @ B
        - A and B are frozen (requires_grad=False) to ensure they receive no gradient from the probe backward pass
        - One forward+backward deposits ∂L/∂W into W_probe.grad

        On exit:
        - W_probe is discarded
        - A, B are restored to requires_grad=True
        - A.data and B.data are completely unchanged

        Usage:
            with layer.grad_probe_mode():
                loss = model(x)
                loss.backward()
            grad_W = layer.get_probe_grad()   # ∂L/∂W, shape [out_ch, in_ch]
        """
        assert self._W_probe is None, "grad_probe_mode() is not re-entrant"

        # Snapshot A, B requires_grad state; freeze them for the probe pass
        a_req = self.A.requires_grad
        b_req = self.B.requires_grad
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)

        # Create the leaf W from the current A, B values — detached so it is its own leaf in the autograd graph, independent of A, B.
        with torch.no_grad():
            W_data = self.A @ self.B                       # [out_ch, in_ch]
        self._W_probe = nn.Parameter(W_data)               # fresh leaf, grad=None

        try:
            yield self                                      # caller runs fwd+bwd
        finally:
            # Restore A, B grad tracking; discard probe
            self.A.requires_grad_(a_req)
            self.B.requires_grad_(b_req)
            self._W_probe = None                           # GC handles cleanup

    def get_probe_grad(self) -> Optional[torch.Tensor]:
        """
        Returns ∂L/∂W from the most recent probe backward pass.

        Must be called INSIDE grad_probe_mode() after backward(),
        or it returns None (no probe is active or backward not yet called).

        Shape: [out_ch, in_ch]
        """
        if self._W_probe is None:
            return None
        return self._W_probe.grad                          # may be None if bwd not run

# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract the live rank_map from any model that contains FactoredConv1d
# ─────────────────────────────────────────────────────────────────────────────

def extract_rank_map(model: nn.Module) -> Dict[str, int]:
    """
    Walk model.named_modules() and collect {module_path: rank} for every FactoredConv1d found.

    This is the only source of truth for ranks — it reads self.rank which FactoredConv1d always keeps in sync with its actual A/B shapes.

    Example output:
        {
            'separator.bottleneck':                    32,
            'separator.conv_layers.0.conv_expand':     16,
            'separator.conv_layers.0.conv_residual':   16,
            ...
        }
    """
    return {
        name: module.rank
        for name, module in model.named_modules()
        if isinstance(module, FactoredConv1d)
    }

# =============================================================================
# SECTION 1: NORMALIZATION LAYERS
# =============================================================================

class CumulativeLayerNorm(nn.Module):
    """
    Cumulative Layer Normalization for causal (real-time) processing.
    
    Why do we need this?
    --------------------
    In real-time audio processing, we can only use information from the PAST,
    not the future. This normalization computes statistics cumulatively,
    meaning at each time step, it only considers samples up to that point.
    
    Parameters
    ----------
    num_features : int -- Number of features/channels in the input.
    eps : float -- Small value to prevent division by zero. Default: 1e-8
    learnable : bool -- If True, learns scaling (gain) and shifting (bias) parameters.
    
    Input Shape: (Batch, Features, Time)
    Output Shape: (Batch, Features, Time) - same as input
    """
    
    def __init__(self, num_features: int, eps: float = 1e-8, learnable: bool = True):
        super().__init__()
        
        self.eps = eps
        self.num_features = num_features
        
        if learnable:
            # Learnable parameters to scale and shift the normalized output
            self.gain = nn.Parameter(torch.ones(1, num_features, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        else:
            # Fixed parameters (no learning)
            self.register_buffer('gain', torch.ones(1, num_features, 1))
            self.register_buffer('bias', torch.zeros(1, num_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cumulative layer normalization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, Features, Time)
        
        Returns
        -------
        torch.Tensor
            Normalized tensor of same shape
        """
        batch_size, num_channels, time_steps = x.size()
        
        # Step 1: Sum across channels for each time step
        step_sum = x.sum(dim=1)  # Shape: (Batch, Time)
        step_squared_sum = x.pow(2).sum(dim=1)  # Shape: (Batch, Time)
        
        # Step 2: Compute cumulative sums (running totals)
        cumulative_sum = torch.cumsum(step_sum, dim=1)  # Shape: (Batch, Time)
        cumulative_squared_sum = torch.cumsum(step_squared_sum, dim=1)
        
        # Step 3: Count how many values we've seen at each time step
        # At time t, we've seen (t+1) * num_channels values
        entry_count = torch.arange(
            num_channels, 
            num_channels * (time_steps + 1), 
            num_channels,
            dtype=x.dtype,
            device=x.device
        ).view(1, -1)  # Shape: (1, Time)
        
        # Step 4: Calculate cumulative mean and standard deviation
        cumulative_mean = cumulative_sum / entry_count
        cumulative_variance = (
            cumulative_squared_sum / entry_count 
            - cumulative_mean.pow(2)
        )
        cumulative_std = (cumulative_variance + self.eps).sqrt()
        
        # Step 5: Normalize (subtract mean, divide by std)
        cumulative_mean = cumulative_mean.unsqueeze(1)  # Add channel dimension
        cumulative_std = cumulative_std.unsqueeze(1)
        
        normalized = (x - cumulative_mean) / cumulative_std
        
        # Step 6: Apply learned scaling and bias
        return normalized * self.gain + self.bias
    
# ─────────────────────────────────────────────────────────────────────────────
# DepthwiseSeparableConv1d  — accepts rank_map + its own prefix
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv1d(nn.Module):

    def __init__(
        self,
        input_channels:  int,
        hidden_channels: int,
        kernel_size:     int,
        dilation:        int  = 1,
        causal:          bool = False,
        use_skip:        bool = True,
        rank_factor:     int  = 4,
        # ── New: rank override ───────────────────────────────────────────── #
        rank_map:        Optional[Dict[str, int]] = None,
        name_prefix:     str = '',          # e.g. 'separator.conv_layers.0'
    ):
        super().__init__()

        self.causal   = causal
        self.use_skip = use_skip

        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2

        rank_map = rank_map or {}

        # ── Helper: rank for a sub-layer of this block ───────────────────── #
        def _rank(attr: str, default_in: int, default_out: int) -> int:
            """
            Look up rank in rank_map by the full dotted path of this attribute.
            Falls back to min(in, out) // rank_factor if not in map.
            """
            full_key = f'{name_prefix}.{attr}' if name_prefix else attr
            return rank_map.get(full_key, min(default_in, default_out) // rank_factor)

        self.conv_expand = FactoredConv1d(
            input_channels, hidden_channels, bias=True,
            rank=_rank('conv_expand', input_channels, hidden_channels),
        )
        self.conv_depthwise = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size, dilation=dilation,
            groups=hidden_channels, padding=self.padding,
        )
        self.conv_residual = FactoredConv1d(
            hidden_channels, input_channels, bias=True,
            rank=_rank('conv_residual', hidden_channels, input_channels),
        )
        if use_skip:
            self.conv_skip = FactoredConv1d(
                hidden_channels, input_channels, bias=True,
                rank=_rank('conv_skip', hidden_channels, input_channels),
            )

        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()

        if causal:
            self.norm1 = CumulativeLayerNorm(hidden_channels)
            self.norm2 = CumulativeLayerNorm(hidden_channels)
        else:
            self.norm1 = nn.GroupNorm(1, hidden_channels, eps=1e-8)
            self.norm2 = nn.GroupNorm(1, hidden_channels, eps=1e-8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process input through the depthwise separable convolution block.
        
        Parameters
        ----------
        x : torch.Tensor -- Input tensor of shape (Batch, Channels, Time)
        
        Returns
        -------
        residual : torch.Tensor -- Output to be added to the main path
        skip : torch.Tensor or None -- Skip connection output (if use_skip=True)
        """
        # First block: expand → activate → normalize
        out = self.norm1(self.activation1(self.conv_expand(x)))
        
        # Second block: depthwise conv → activate → normalize
        out = self.conv_depthwise(out)
        
        # If causal, remove the "future" samples we padded
        if self.causal and self.padding > 0:
            out = out[:, :, :-self.padding]
        
        out = self.norm2(self.activation2(out))
        
        # Create outputs
        residual = self.conv_residual(out)
        
        if self.use_skip:
            skip = self.conv_skip(out)
            return residual, skip
        else:
            return residual, None

# ─────────────────────────────────────────────────────────────────────────────
# TemporalConvNet  — passes rank_map + computed prefix down to each block
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConvNet(nn.Module):

    def __init__(
        self,
        input_dim:      int,
        output_dim:     int,
        bottleneck_dim: int,
        hidden_dim:     int,
        num_layers:     int,
        num_stacks:     int,
        kernel_size:    int  = 3,
        causal:         bool = False,
        use_skip:       bool = True,
        rank_factor:    int  = 4,
        # ── New: rank override ───────────────────────────────────────────── #
        rank_map:       Optional[Dict[str, int]] = None,
        name_prefix:    str = 'separator',   # matches where TCN sits in ConvTasNet
    ):
        super().__init__()

        self.use_skip = use_skip
        rank_map      = rank_map or {}

        if causal:
            self.input_norm = CumulativeLayerNorm(input_dim)
        else:
            self.input_norm = nn.GroupNorm(1, input_dim, eps=1e-8)

        # Bottleneck FactoredConv1d
        bottleneck_key  = f'{name_prefix}.bottleneck'
        bottleneck_rank = rank_map.get(
            bottleneck_key,
            min(input_dim, bottleneck_dim) // rank_factor,
        )
        self.bottleneck = FactoredConv1d(
            input_dim, bottleneck_dim,
            rank=bottleneck_rank, bias=True,
        )

        # Conv layers
        self.conv_layers    = nn.ModuleList()
        self.receptive_field = 0

        for stack_idx in range(num_stacks):
            for layer_idx in range(num_layers):
                dilation   = 2 ** layer_idx
                layer_path = f'{name_prefix}.conv_layers.{stack_idx * num_layers + layer_idx}'

                self.conv_layers.append(
                    DepthwiseSeparableConv1d(
                        input_channels  = bottleneck_dim,
                        hidden_channels = hidden_dim,
                        kernel_size     = kernel_size,
                        dilation        = dilation,
                        causal          = causal,
                        use_skip        = use_skip,
                        rank_factor     = rank_factor,
                        rank_map        = rank_map,
                        name_prefix     = layer_path,   # e.g. 'separator.conv_layers.3'
                    )
                )

                if stack_idx == 0 and layer_idx == 0:
                    self.receptive_field += kernel_size
                else:
                    self.receptive_field += (kernel_size - 1) * dilation

        # Output FactoredConv1d  (lives at index [1] inside nn.Sequential)
        output_key  = f'{name_prefix}.output_layer.1'
        output_rank = rank_map.get(
            output_key,
            min(bottleneck_dim, output_dim) // rank_factor,
        )
        self.output_layer = nn.Sequential(
            nn.PReLU(),
            FactoredConv1d(bottleneck_dim, output_dim, rank=output_rank, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process encoded audio features to create separation masks.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, Features, Time)
        
        Returns
        -------
        torch.Tensor
            Output tensor (mask) of shape (Batch, Output_Features, Time)
        """
        # Normalize and reduce to bottleneck dimension
        out = self.bottleneck(self.input_norm(x))
        
        # Process through all conv layers
        if self.use_skip:
            skip_sum = 0.0
            for layer in self.conv_layers:
                residual, skip = layer(out)
                out = out + residual
                skip_sum = skip_sum + skip
            out = self.output_layer(skip_sum)
        else:
            for layer in self.conv_layers:
                residual, _ = layer(out)
                out = out + residual
            out = self.output_layer(out)
        
        return out
    
# ─────────────────────────────────────────────────────────────────────────────
# ConvTasNet  — gets rank_map, passes it to TemporalConvNet, adds from_checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class ConvTasNet(nn.Module):

    def __init__(
        self,
        num_sources:  int   = 2,
        encoder_dim:  int   = 512,
        feature_dim:  int   = 128,
        sample_rate:  int   = 8000,
        window_ms:    float = 2.0,
        num_layers:   int   = 8,
        num_stacks:   int   = 3,
        kernel_size:  int   = 3,
        causal:       bool  = False,
        use_skip:     bool  = True,
        rank_factor:  int   = 4,
        # ── New: rank override — None means use rank_factor defaults ──────── #
        rank_map:     Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        # ── Store all hparams so from_checkpoint can reconstruct cleanly ─── #
        self.hparams = dict(
            num_sources = num_sources,
            encoder_dim = encoder_dim,
            feature_dim = feature_dim,
            sample_rate = sample_rate,
            window_ms   = window_ms,
            num_layers  = num_layers,
            num_stacks  = num_stacks,
            kernel_size = kernel_size,
            causal      = causal,
            use_skip    = use_skip,
            rank_factor = rank_factor,
        )

        self.num_sources = num_sources
        self.encoder_dim = encoder_dim
        self.window_size = int(sample_rate * window_ms / 1000)
        self.hop_size    = self.window_size // 2

        self.encoder = nn.Conv1d(
            1, encoder_dim,
            kernel_size=self.window_size,
            stride=self.hop_size,
            bias=False,
        )

        self.separator = TemporalConvNet(
            input_dim      = encoder_dim,
            output_dim     = encoder_dim * num_sources,
            bottleneck_dim = feature_dim,
            hidden_dim     = feature_dim * 4,
            num_layers     = num_layers,
            num_stacks     = num_stacks,
            kernel_size    = kernel_size,
            causal         = causal,
            use_skip       = use_skip,
            rank_factor    = rank_factor,
            rank_map       = rank_map,          # ← propagated here
            name_prefix    = 'separator',
        )

        self.receptive_field = self.separator.receptive_field

        self.decoder = nn.ConvTranspose1d(
            encoder_dim, 1,
            kernel_size=self.window_size,
            stride=self.hop_size,
            bias=False,
        )

    # ── Class method: reconstruct model at checkpoint ranks ──────────────── #

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device:          str = 'cpu',
    ) -> Tuple['ConvTasNet', dict]:
        """
        Build a ConvTasNet whose FactoredConv1d layers are sized to exactly match the saved checkpoint, then load the weights.
        
        Parameters
        ----------
        checkpoint_path : str  — path to a .pth file saved by Trainer
        device          : str  — 'cpu' | 'cuda' | 'cuda:0' etc.

        Returns
        -------
        model      : ConvTasNet  — ready for training or inference
        checkpoint : dict        — full checkpoint dict (optimizer_state_dict etc.)

        Flow
        ----
        1. Load checkpoint dict (always to CPU first — safe regardless of origin)
        2. Read hparams  → guarantees same architecture hyperparameters
        3. Read rank_map → each FactoredConv1d is allocated at its saved rank
        4. Instantiate model  (shapes now match the state_dict exactly)
        5. load_state_dict(strict=True)  — no shape surgery needed
        6. Move to device
        """
        raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        hparams  = raw['hparams']
        rank_map = raw.get('rank_map', {})   # empty = fresh model, use defaults

        # Build model at the exact saved ranks
        model = cls(**hparams, rank_map=rank_map)

        # strict=True is safe here — shapes are guaranteed to match
        missing, unexpected = model.load_state_dict(
            raw['model_state_dict'], strict=True
        )
        # These should always be empty; surface them loudly if not
        if missing or unexpected:
            raise RuntimeError(
                f'State dict mismatch after rank-aware construction.\n'
                f'  Missing:    {missing}\n'
                f'  Unexpected: {unexpected}\n'
                f'This should never happen — please file a bug.'
            )

        model.to(device)
        print(
            f'[ConvTasNet] Loaded from {checkpoint_path}\n'
            f'             {len(rank_map)} factored layers restored.'
        )
        return model, raw

    def _pad_signal(self, audio: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Pad the input audio to ensure proper encoding/decoding.
        
        Why padding?
        -----------
        The encoder window needs to align properly with the signal.
        We also add padding at the beginning/end to avoid edge effects.
        
        Parameters
        audio : torch.Tensor -- Input audio of shape (Batch, Time) or (Batch, 1, Time)
        
        Returns
        padded_audio : torch.Tensor -- Padded audio of shape (Batch, 1, PaddedTime)
        padding_amount : int -- Amount of padding added at the end (needed for unpadding later)
        """
        # Ensure 3D input: (Batch, Channels, Time)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() != 3:
            raise ValueError(f"Input must be 2D or 3D, got {audio.dim()}D")
        
        batch_size, _, num_samples = audio.size()
        
        # Calculate padding needed to align with window/hop
        remainder = (self.hop_size + num_samples % self.window_size) % self.window_size
        padding_end = self.window_size - remainder if remainder > 0 else 0
        
        # Pad the end if needed
        if padding_end > 0:
            audio = F.pad(audio, (0, padding_end))
        
        # Add padding at beginning and end for edge effects
        audio = F.pad(audio, (self.hop_size, self.hop_size))
        
        return audio, padding_end

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Separate a mixed audio signal into individual sources.
        
        Parameters
        ----------
        mixture : torch.Tensor -- Mixed audio signal of shape (Batch, Time) or (Batch, 1, Time)
        
        Returns
        -------
        torch.Tensor -- Separated sources of shape (Batch, NumSources, Time)
            
        Example
        -------
        >>> model = ConvTasNet(num_sources=2)
        >>> mix = torch.randn(1, 16000)  # 2 seconds of audio
        >>> sources = model(mix)  # Shape: (1, 2, 16000)
        """
        # Step 1: Pad the input signal
        padded, padding_amount = self._pad_signal(mixture)
        batch_size = padded.size(0)
        
        # Step 2: ENCODE - Transform waveform to latent representation
        # Shape: (Batch, EncoderDim, TimeFrames)
        encoded = self.encoder(padded)
        
        # Step 3: SEPARATE - Generate masks using TCN
        # Shape: (Batch, EncoderDim * NumSources, TimeFrames)
        mask_output = self.separator(encoded)
        
        # Apply sigmoid to get masks between 0 and 1
        # Reshape to (Batch, NumSources, EncoderDim, TimeFrames)
        masks = torch.sigmoid(mask_output).view(
            batch_size, self.num_sources, self.encoder_dim, -1
        )
        
        # Step 4: Apply masks to encoded representation
        # Multiply encoded by each mask to isolate sources
        # encoded: (Batch, EncoderDim, Time) → unsqueeze → (Batch, 1, EncoderDim, Time)
        masked = encoded.unsqueeze(1) * masks  # (Batch, NumSources, EncoderDim, Time)
        
        # Step 5: DECODE - Transform back to waveforms
        # Process all sources at once by combining batch and source dimensions
        masked_flat = masked.view(batch_size * self.num_sources, self.encoder_dim, -1)
        decoded = self.decoder(masked_flat)  # (Batch*NumSources, 1, Time)
        
        # Step 6: Remove padding and reshape
        # Calculate where to trim
        start = self.hop_size
        end = -(padding_amount + self.hop_size) if padding_amount > 0 else -self.hop_size
        
        output = decoded[:, :, start:end].contiguous()
        output = output.view(batch_size, self.num_sources, -1)
        
        return output
    
# =============================================================================
# SECTION 4: LOSS FUNCTIONS
# =============================================================================
class SI_SNR_Loss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss.
    
    Overview
    --------
    SI-SNR measures separation quality while ignoring volume differences.
    It answers: "How much of the estimate is signal vs. noise?"
    
    Mathematical Formulation
    ------------------------
    Given estimated signal ŝ and target signal s:
    
        1. Optimal scaling:     α = <ŝ, s> / ||s||²
        2. Signal component:    s_target = α · s
        3. Noise component:     e_noise = ŝ - s_target
        4. SI-SNR (dB):         10 · log₁₀(||s_target||² / ||e_noise||²)
    
    Interpretation
    --------------
        SI-SNR (dB)  │  Quality
        ─────────────┼──────────────
           < 0       │  Very poor (more noise than signal)
           0 - 5     │  Poor
           5 - 10    │  Acceptable
          10 - 15    │  Good
          15 - 20    │  Very good
           > 20      │  Excellent
    
    Training Note
    -------------
    We return NEGATIVE SI-SNR so that minimizing loss = maximizing SI-SNR.
    """
    
    def __init__(self, zero_mean: bool = True, eps: float = 1e-8):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps
    
    def forward(
        self, 
        estimated: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SI-SNR loss (per-sample, not averaged).
        
        Parameters
        ----------
        estimated : torch.Tensor
            Estimated signals, shape (batch_size, num_samples)
        target : torch.Tensor
            Target signals, shape (batch_size, num_samples)
        
        Returns
        -------
        torch.Tensor
            Negative SI-SNR for each sample, shape (batch_size,)
            Lower values indicate better separation.
        """
        # ─────────────────────────────────────────────────────────────────────
        # Step 0: Remove DC offset (mean) for true scale invariance
        # ─────────────────────────────────────────────────────────────────────
        if self.zero_mean:
            estimated = estimated - estimated.mean(dim=-1, keepdim=True)
            target = target - target.mean(dim=-1, keepdim=True)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Compute optimal scaling factor α
        # 
        #         <ŝ, s>        Σᵢ ŝᵢ · sᵢ
        #    α = ────────  =  ──────────────
        #         ||s||²         Σᵢ sᵢ²
        #
        # This projects ŝ onto s to find the best-fit scaling.
        # ─────────────────────────────────────────────────────────────────────
        dot_product = torch.sum(estimated * target, dim=-1, keepdim=True)
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        scaling_factor = dot_product / target_energy
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Decompose estimate into signal and noise
        #
        #    s_target = α · s           (signal component)
        #    e_noise  = ŝ - s_target    (everything else = noise)
        #
        #    ŝ ──────┬────────────► s_target (aligned with target)
        #            │
        #            └────────────► e_noise  (orthogonal residual)
        # ─────────────────────────────────────────────────────────────────────
        signal_component = scaling_factor * target
        noise_component = estimated - signal_component
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Compute SI-SNR in decibels
        #
        #                    ||s_target||²
        #    SI-SNR = 10 · log₁₀ ──────────────
        #                    ||e_noise||²
        # ─────────────────────────────────────────────────────────────────────
        signal_energy = torch.sum(signal_component ** 2, dim=-1) + self.eps
        noise_energy = torch.sum(noise_component ** 2, dim=-1) + self.eps
        
        si_snr_db = 10 * torch.log10(signal_energy / noise_energy)
        
        # Return negative for loss minimization (we want to MAXIMIZE SI-SNR)
        return -si_snr_db  # Shape: (batch_size,)


class PIT_SI_SNR_Loss(nn.Module):
    """
    Permutation Invariant Training (PIT) with SI-SNR Loss.
    
    The Problem
    -----------
    In source separation, model outputs have no inherent ordering:
    
        Model outputs:  [output_0, output_1]
        True sources:   [speaker_A, speaker_B]
        
        Which output corresponds to which speaker? We don't know!
    
    The Solution (PIT)
    ------------------
    Try ALL possible assignments and use the one with lowest loss:
    
        ┌─────────────────────────────────────────────────────────┐
        │  Permutation 1:  output_0 → speaker_A                   │
        │                  output_1 → speaker_B    →  loss = 2.3  │
        ├─────────────────────────────────────────────────────────┤
        │  Permutation 2:  output_0 → speaker_B                   │
        │                  output_1 → speaker_A    →  loss = 5.1  │
        └─────────────────────────────────────────────────────────┘
                                    ↓
                    Select Permutation 1 (lower loss)
    
    Per-Sample Selection
    --------------------
    IMPORTANT: Each sample in a batch gets its OWN best permutation!
    
        Sample 0: Best = perm (0,1)  ┐
        Sample 1: Best = perm (1,0)  │  Different samples,
        Sample 2: Best = perm (0,1)  │  different best perms!
        Sample 3: Best = perm (1,0)  ┘
    
    Complexity Note
    ---------------
    Number of permutations = n! where n = number of sources
    
        Sources │ Permutations
        ────────┼─────────────
           2    │      2
           3    │      6
           4    │     24
           5    │    120  (gets expensive!)
    
    Parameters
    ----------
    zero_mean : bool, default=True -- Use zero-mean SI-SNR computation.
    """
    
    def __init__(self, zero_mean: bool = True):
        super().__init__()
        self.si_snr_loss = SI_SNR_Loss(zero_mean=zero_mean)

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        estimated : (Batch, NumSources, Samples)
        target    : (Batch, NumSources, Samples)

        Returns
        -------
        loss : scalar (batch-averaged best losses)
        """
        batch_size, num_sources, num_samples = estimated.size()
        perms = list(permutations(range(num_sources)))

        # Store per-sample loss for each permutation
        # Shape: (num_perms, batch_size)
        perm_losses = []

        for perm in perms:
            perm_estimated = estimated[:, perm, :]

            # Sum SI-SNR across sources for each sample
            # Shape: (batch_size,)
            perm_loss = torch.zeros(batch_size, device=estimated.device)
            for src_idx in range(num_sources):
                perm_loss = perm_loss + self.si_snr_loss(
                    perm_estimated[:, src_idx, :],
                    target[:, src_idx, :]
                )

            perm_losses.append(perm_loss / num_sources)

        # Stack: (num_perms, batch_size)
        perm_losses = torch.stack(perm_losses, dim=0)

        # Find best permutation for EACH sample independently
        # min_losses shape: (batch_size,)
        # best_perm_idx shape: (batch_size,)
        min_losses, best_perm_idx = perm_losses.min(dim=0)

        # Average across batch
        return min_losses.mean()
    
def _svd_of_product(
    A: torch.Tensor,   # [n, k]
    B: torch.Tensor,   # [k, m]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Economy SVD of W = A @ B via a QR-based shortcut.

    Steps
    -----
    1. QR-decompose A   = Q_A R_A    →  O(n k²)
    2. QR-decompose Bᵀ  = Q_B R_B    →  O(m k²)
    3. SVD of (R_A Rᵀ_B)             →  O(k³)   only kxk step
    4. U = Q_A U_s,   Vt = Vᵀ_s Qᵀ_B

    Avoids forming the full [n, m] product; cost dominated by O(k³) not O(nm).

    Returns
    -------
    U  : [n, k]   left singular vectors
    S  : [k]      singular values, descending
    Vt : [k, m]   right singular vectors (transposed)
    """
    Q_A, R_A = torch.linalg.qr(A,   mode='reduced')   # [n,k], [k,k]
    Q_B, R_B = torch.linalg.qr(B.T, mode='reduced')   # [m,k], [k,k]
    U_s, S, Vt_s = torch.linalg.svd(R_A @ R_B.T, full_matrices=False)
    return Q_A @ U_s, S, Vt_s @ Q_B.T                 # [n,k], [k], [k,m]

@torch.no_grad()
def rebalance_all_factored_layers(model: nn.Module) -> int:
    """
    Re-express every FactoredConv1d layer's A, B in the SVD basis of W = A @ B.

    What this does
    --------------
    For each FactoredConv1d layer, the effective weight is W = A @ B.
    Over the course of training, A and B can become poorly scaled relative
    to each other — one factor dominates the singular values while the other
    shrinks.  Re-balancing re-writes A, B as:

        U, S, Vt = SVD(A @ B)
        A_new = U  * sqrt(S)     [out_ch, rank]
        B_new = Vt * sqrt(S)     [rank,   in_ch]

    so that A_new @ B_new = W exactly, but the singular-value energy is now split equally between both factors.
    """
    n_rebalanced = 0

    for name, module in model.named_modules():
        if not isinstance(module, FactoredConv1d):
            continue

        A = module.A.data   # [out_ch, rank]
        B = module.B.data   # [rank,   in_ch]

        # QR-based economy SVD of W = A @ B  (avoids forming the full matrix)
        U, S, Vt = _svd_of_product(A, B)

        sqrt_S = S.sqrt()                          # [rank]
        module.A.data.copy_((U  * sqrt_S.unsqueeze(0)).contiguous())  # [out_ch, rank]
        module.B.data.copy_((Vt * sqrt_S.unsqueeze(1)).contiguous())  # [rank,   in_ch]

        n_rebalanced += 1

    return n_rebalanced
    
# =============================================================================
# SECTION 5: TRAINING SCRIPT
# =============================================================================

class Trainer:
    """
    Training manager for Conv-TasNet.
    
    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging to TensorBoard
    - Learning rate scheduling  # ← ADDED to docstring
    
    Parameters
    ----------
    model : nn.Module -- Conv-TasNet model
    train_loader : DataLoader -- Training data loader
    val_loader : DataLoader -- Validation data loader
    optimizer : torch.optim.Optimizer -- Optimizer (e.g., Adam)
    scheduler : torch.optim.lr_scheduler._LRScheduler -- Learning rate scheduler (optional)  # ← ADDED
    device : str -- Device to train on ('cuda' or 'cpu')
    checkpoint_dir : str -- Directory to save checkpoints
    log_dir : str -- Directory for TensorBoard logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,  # ← ADDED parameter
        rebalance_every_epoch: bool = False,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler  # ← ADDED: Store the scheduler
        self.rebalance_every_epoch = rebalance_every_epoch
        self.device = device
        
        # Loss function
        self.criterion = PIT_SI_SNR_Loss(zero_mean=True)
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns
        -------
        avg_loss : float -- Average training loss
        avg_si_snr : float -- Average SI-SNR in dB
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (mixture, sources) in enumerate(self.train_loader):
            # Move to device
            mixture = mixture.to(self.device)  # (Batch, Samples)
            sources = sources.to(self.device)  # (Batch, NumSources, Samples)
            
            # Forward pass
            estimated = self.model(mixture)  # (Batch, NumSources, Samples)
            
            # Calculate loss
            loss = self.criterion(estimated, sources)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (helps with stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Log to TensorBoard
            if batch_idx % 500 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
            
            self.global_step += 1
            
            # Print progress
            if batch_idx % 1000 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns
        -------
        avg_loss : float -- Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        for mixture, sources in self.val_loader:
            mixture = mixture.to(self.device)
            sources = sources.to(self.device)
            
            # Forward pass
            estimated = self.model(mixture)
            
            # Calculate loss
            loss = self.criterion(estimated, sources)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """
        Save a rank-aware checkpoint.

        Additions vs. original:
          'hparams'  — architecture hyperparameters (num_sources, encoder_dim …)
          'rank_map' — {module_path: int} current rank of every FactoredConv1d

        Both are derived directly from the live model — zero manual bookkeeping.
        They are used together by ConvTasNet.from_checkpoint() to reconstruct
        the model at the exact same shapes before loading the state dict.
        """
        checkpoint = {
            'epoch':               self.epoch,
            'global_step':         self.global_step,
            'best_val_loss':       self.best_val_loss,
            'model_state_dict':    self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # ── New fields ──────────────────────────────────────────────── #
            'hparams':  self.model.hparams,          # fixed architecture params
            'rank_map': extract_rank_map(self.model), # live per-layer ranks
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)
        rank_map = checkpoint['rank_map']
        print(
            f'  ✓ Checkpoint saved: {filename}  '
            f'({len(rank_map)} factored layers, '
            f'ranks {min(rank_map.values())}-{max(rank_map.values())})'
        )

    def load_checkpoint(self, filename: str = 'checkpoint.pth') -> bool:
        """
        Load a rank-aware checkpoint.

        The new approach:
            1. Rebuild self.model IN-PLACE at saved ranks via from_checkpoint
            2. Rebuild optimizer against the new model.parameters()
            3. Restore optimizer state
            4. Restore scheduler state

        self.model is reassigned — the Trainer always holds the correct model.
        """
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            print(f'  ✗ Checkpoint not found: {filename}')
            return False

        # ── 1. Rebuild model at saved ranks + load weights ────────────────── #
        model, raw = ConvTasNet.from_checkpoint(
            checkpoint_path = str(checkpoint_path),
            device          = self.device,
        )
        self.model = model  # replace the Trainer's model reference

        # ── 2. Restore training state ─────────────────────────────────────── #
        self.epoch         = raw['epoch']
        self.global_step   = raw['global_step']
        self.best_val_loss = raw['best_val_loss']

        # ── 3. Rebuild optimizer against the newly constructed model ──────── #
        saved_opt   = raw['optimizer_state_dict']
        saved_lr    = saved_opt['param_groups'][0]['lr']

        self.optimizer = type(self.optimizer)(
            self.model.parameters(), lr=saved_lr
        )
        self.optimizer.load_state_dict(saved_opt)

        # Move optimizer state tensors to the correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # ── 4. Restore scheduler if present ──────────────────────────────── #
        if self.scheduler is not None:
            self.scheduler.optimizer = self.optimizer              # ← THE FIX
            
            if 'scheduler_state_dict' in raw:
                self.scheduler.load_state_dict(raw['scheduler_state_dict'])

        print(f'  ✓ Checkpoint loaded: {filename} (epoch {self.epoch})')
        return True
    
    def _get_current_lr(self) -> float:  # ← ADDED: Helper method to get current learning rate
        """Get the current learning rate from the optimizer."""
        return self.optimizer.param_groups[0]['lr']
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if not checkpoints:
            return None
        
        # Extract epoch numbers and find the latest
        def get_epoch_num(path):
            # Extract number from 'checkpoint_epoch_X.pth'
            name = path.stem  # 'checkpoint_epoch_X'
            return int(name.split('_')[-1])
        
        latest = max(checkpoints, key=get_epoch_num)
        return latest.name

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Parameters
        ----------
        num_epochs : int -- Total number of epochs to train
        resume_from : str, optional -- Checkpoint filename to resume from
        """
        # ─────────────────────────────────────────────────────────────────────
        # Handle checkpoint resumption
        # ─────────────────────────────────────────────────────────────────────
        start_epoch = 0
        
        if resume_from is not None:
            if self.load_checkpoint(resume_from):
                start_epoch = self.epoch + 1  # Start from NEXT epoch
                print(f"  ▶ Resuming from epoch {start_epoch}")
            else:
                print(f"  ⚠ Could not load checkpoint, starting from scratch")
        
        print("=" * 70)
        print("TRAINING CONV-TASNET")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total Epochs: {num_epochs}")
        print(f"Starting Epoch: {start_epoch + 1}")  # Human-readable (1-indexed)
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Current LR: {self._get_current_lr():.6f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print("=" * 70)
        
        # ─────────────────────────────────────────────────────────────────────
        # Training loop - START FROM start_epoch, NOT 0
        # ─────────────────────────────────────────────────────────────────────
        if start_epoch == 0:
            self.save_checkpoint(f'checkpoint_epoch_{start_epoch}.pth')  # Save initial checkpoint

        for epoch in range(start_epoch, num_epochs):  # ← Changed!
            self.epoch = epoch
            epoch_start_time = time.time()
            
            current_lr = self._get_current_lr()
            print(f"\nEpoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})")
            print("-" * 70)

            # ── Per-epoch SVD rebalance (before any gradient steps) ────────── #
            # ← NEW BLOCK START
            if self.rebalance_every_epoch:
                n = rebalance_all_factored_layers(self.model)
                print(f'  [Rebalance] {n} FactoredConv1d layers rebalanced.')
            # ← NEW BLOCK END
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            epoch_time = time.time() - epoch_start_time
            
            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log to TensorBoard (uses epoch as x-axis, so logs continue correctly)
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self._get_current_lr(), epoch)
            
            # Print summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self._get_current_lr():.6f}")
            print(f"  Time:       {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"  🌟 New best model!")

            # Save checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print(f"Final LR: {self._get_current_lr():.6f}")
        print("=" * 70)
        self.writer.close()

class PairedAudioDataset(Dataset):
    """
    Same as PairedAudioDataset but with safety checks for edge cases.
    Use this if you're not 100% sure all files are exactly 4 seconds.
    """
    
    def __init__(
        self,
        mixed_dir: str,
        source_dirs: list,
        sample_rate: int = 8000,
        expected_duration: float = 4.0
    ):
        self.mixed_dir = mixed_dir
        self.source_dirs = source_dirs
        self.sample_rate = sample_rate
        self.expected_length = int(sample_rate * expected_duration)
        
        # Get files that exist in ALL directories
        mixed_files = set(f for f in os.listdir(mixed_dir) if f.endswith('.wav'))
        
        for src_dir in source_dirs:
            src_files = set(f for f in os.listdir(src_dir) if f.endswith('.wav'))
            mixed_files = mixed_files.intersection(src_files)
        
        self.file_list = sorted(list(mixed_files))
        print(f"📁 Found {len(self.file_list)} matching audio files")
    
    def __len__(self):
        return len(self.file_list)
    
    def _ensure_length(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is exactly the expected length."""
        if len(audio) > self.expected_length:
            # Trim if too long
            return audio[:self.expected_length]
        elif len(audio) < self.expected_length:
            # Pad if too short
            pad_len = self.expected_length - len(audio)
            return np.pad(audio, (0, pad_len), mode='constant')
        return audio
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # Load mixed audio
        mixed_path = os.path.join(self.mixed_dir, filename)
        mixed_audio, _ = sf.read(mixed_path)
        mixed_audio = self._ensure_length(mixed_audio)
        
        # Load source audio files
        source_audios = []
        for src_dir in self.source_dirs:
            src_path = os.path.join(src_dir, filename)
            src_audio, _ = sf.read(src_path)
            src_audio = self._ensure_length(src_audio)
            source_audios.append(src_audio)
        
        # Convert to tensors
        mixed_tensor = torch.tensor(mixed_audio, dtype=torch.float32).unsqueeze(0)
        sources_tensor = torch.tensor(np.stack(source_audios), dtype=torch.float32)
        
        return mixed_tensor, sources_tensor
    
def main(resume: bool = False):
    config = {
        # Model parameters
        'num_sources': 2,
        'encoder_dim': 512,
        'feature_dim': 128,
        'sample_rate': 8000,
        'window_ms': 2,
        'num_layers': 8,
        'num_stacks': 3,
        'kernel_size': 3,
        'causal': False,
        'use_skip': True,
        
        # Training parameters
        'batch_size': 6,
        'num_epochs': 50,
        'learning_rate': 0.000125,
        'weight_decay': 1e-5,
        
        # System parameters
        'device': 'cuda',
        'num_workers': 0,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
    }

    # Initialize model
    model = ConvTasNet(
        num_sources=config['num_sources'],
        encoder_dim=config['encoder_dim'],
        feature_dim=config['feature_dim'],
        sample_rate=config['sample_rate'],
        window_ms=config['window_ms'],
        num_layers=config['num_layers'],
        num_stacks=config['num_stacks'],
        kernel_size=config['kernel_size'],
        causal=config['causal'],
        use_skip=config['use_skip']
    ).to(config['device'])
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
   # For training
    # Load mixed and source audio
    mixed_dir_train = "D:\Datasets\Source_Separation_Dataset\Data\LibriSpeech\Libri2Mix_4s\\train-clean-100_4s\mix_clean"
    source_dirs_train = ["D:\Datasets\Source_Separation_Dataset\Data\LibriSpeech\Libri2Mix_4s\\train-clean-100_4s\s1", \
                        "D:\Datasets\Source_Separation_Dataset\Data\LibriSpeech\Libri2Mix_4s\\train-clean-100_4s\s2"]  # List of source directories
    train__dataset = PairedAudioDataset(
        mixed_dir=mixed_dir_train,
        source_dirs=source_dirs_train,
        sample_rate=config['sample_rate'],
        expected_duration=4.0
    )
    
    train_loader = DataLoader(
            train__dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    # For validation
    # Load mixed and source audio
    mixed_dir_val = 'D:\Datasets\Source_Separation_Dataset\Data\LibriSpeech\Libri2Mix_4s\dev_clean_4s\mix_clean'
    source_dirs_val = ['D:\Datasets\Source_Separation_Dataset\Data\LibriSpeech\Libri2Mix_4s\dev_clean_4s\s1', \
                    'D:\Datasets\Source_Separation_Dataset\Data\LibriSpeech\Libri2Mix_4s\dev_clean_4s\s2']  # List of source directories
    val_dataset = PairedAudioDataset(
        mixed_dir=mixed_dir_val,
        source_dirs=source_dirs_val,
        sample_rate=config['sample_rate'],
        expected_duration=4.0
    )
    val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,  # Validation should not shuffle
            num_workers=config['num_workers'],
            pin_memory=True
        )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,  # ← ADDED: Pass the scheduler to the trainer
        rebalance_every_epoch = True,
        device=config['device'],
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir']
    )
    
    # Check for existing checkpoint if resuming
    resume_checkpoint = None
    if resume:
        resume_checkpoint = trainer.find_latest_checkpoint()
        if resume_checkpoint:
            print(f"Found checkpoint: {resume_checkpoint}")
        else:
            print("No checkpoint found, starting fresh")
    
    # Start training
    trainer.train(
        num_epochs=config['num_epochs'],
        resume_from=resume_checkpoint  # ← Pass checkpoint name
    )

if __name__ == "__main__":
    main(resume=True)