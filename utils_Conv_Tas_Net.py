import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import permutations
from typing import Optional, Tuple
from pathlib import Path
import time
import os
import soundfile as sf

# =============================================================================
# SECTION 1: NORMALIZATION LAYERS
# =============================================================================

class CumulativeLayerNorm(nn.Module):
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
    
# =============================================================================
# SECTION 2: TEMPORAL CONVOLUTIONAL NETWORK (TCN) COMPONENTS
# =============================================================================

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self, 
        input_channels: int, 
        hidden_channels: int, 
        kernel_size: int,
        dilation: int = 1,
        causal: bool = False,
        use_skip: bool = True
    ):
        super().__init__()
        
        self.causal = causal
        self.use_skip = use_skip
        
        # Calculate padding
        if causal:
            # For causal: pad only on the left side
            self.padding = (kernel_size - 1) * dilation
        else:
            # For non-causal: pad equally on both sides
            self.padding = ((kernel_size - 1) * dilation) // 2
        
        # Layer 1: Pointwise (1x1) convolution to expand channels
        self.conv_expand = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        
        # Layer 2: Depthwise convolution (each channel processed independently)
        self.conv_depthwise = nn.Conv1d(
            hidden_channels, 
            hidden_channels, 
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,  # This makes it "depthwise"
            padding=self.padding
        )
        
        # Layer 3: Pointwise convolution to create residual output
        self.conv_residual = nn.Conv1d(hidden_channels, input_channels, kernel_size=1)
        
        # Optional: Skip connection output
        if use_skip:
            self.conv_skip = nn.Conv1d(hidden_channels, input_channels, kernel_size=1)
        
        # Activation functions (PReLU learns the slope for negative values)
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        
        # Normalization layers
        if causal:
            self.norm1 = CumulativeLayerNorm(hidden_channels)
            self.norm2 = CumulativeLayerNorm(hidden_channels)
        else:
            # GroupNorm with 1 group = LayerNorm equivalent for conv layers
            self.norm1 = nn.GroupNorm(1, hidden_channels, eps=1e-8)
            self.norm2 = nn.GroupNorm(1, hidden_channels, eps=1e-8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        
class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_stacks: int,
        kernel_size: int = 3,
        causal: bool = False,
        use_skip: bool = True
    ):
        super().__init__()
        
        self.use_skip = use_skip
        
        # Input normalization
        if causal:
            self.input_norm = CumulativeLayerNorm(input_dim)
        else:
            self.input_norm = nn.GroupNorm(1, input_dim, eps=1e-8)
        
        # Bottleneck layer: reduce dimensions for efficiency
        self.bottleneck = nn.Conv1d(input_dim, bottleneck_dim, kernel_size=1)
        
        # Build the stack of dilated convolution layers
        self.conv_layers = nn.ModuleList()
        self.receptive_field = 0
        
        for stack_idx in range(num_stacks):
            for layer_idx in range(num_layers):
                # Dilation doubles each layer: 1, 2, 4, 8, 16, ...
                dilation = 2 ** layer_idx
                
                self.conv_layers.append(
                    DepthwiseSeparableConv1d(
                        input_channels=bottleneck_dim,
                        hidden_channels=hidden_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        causal=causal,
                        use_skip=use_skip
                    )
                )
                
                # Track receptive field (how far back we can "see")
                if stack_idx == 0 and layer_idx == 0:
                    self.receptive_field += kernel_size
                else:
                    self.receptive_field += (kernel_size - 1) * dilation
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bottleneck_dim, output_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
# =============================================================================
# SECTION 3: MAIN MODEL - Conv-TasNet
# =============================================================================

class ConvTasNet(nn.Module):
    def __init__(
        self,
        num_sources: int = 2,
        encoder_dim: int = 512,
        feature_dim: int = 128,
        sample_rate: int = 8000,
        window_ms: float = 2,
        num_layers: int = 8,
        num_stacks: int = 3,
        kernel_size: int = 3,
        causal: bool = False,
        use_skip: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.num_sources = num_sources
        self.encoder_dim = encoder_dim
        
        # Calculate window size in samples
        # Example: 2ms at 8000Hz = 0.002 * 8000 = 16 samples
        self.window_size = int(sample_rate * window_ms / 1000)
        self.hop_size = self.window_size // 2  # 50% overlap
        
        # === ENCODER ===
        # Converts waveform to learned representation
        # Like computing a spectrogram, but with learned filters
        self.encoder = nn.Conv1d(
            in_channels=1,              # Mono audio
            out_channels=encoder_dim,   # Number of "frequency bins"
            kernel_size=self.window_size,
            stride=self.hop_size,       # Hop between windows
            bias=False
        )
        
        # === SEPARATOR (TCN) ===
        # Creates masks to separate sources
        self.separator = TemporalConvNet(
            input_dim=encoder_dim,
            output_dim=encoder_dim * num_sources,  # One mask per source
            bottleneck_dim=feature_dim,
            hidden_dim=feature_dim * 4,
            num_layers=num_layers,
            num_stacks=num_stacks,
            kernel_size=kernel_size,
            causal=causal,
            use_skip=use_skip
        )
        
        # Store receptive field for reference
        self.receptive_field = self.separator.receptive_field
        
        # === DECODER ===
        # Converts masked representation back to waveform
        self.decoder = nn.ConvTranspose1d(
            in_channels=encoder_dim,
            out_channels=1,
            kernel_size=self.window_size,
            stride=self.hop_size,
            bias=False
        )

    def _pad_signal(self, audio: torch.Tensor) -> Tuple[torch.Tensor, int]:
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
        
        if self.zero_mean:
            estimated = estimated - estimated.mean(dim=-1, keepdim=True)
            target = target - target.mean(dim=-1, keepdim=True)
        
        dot_product = torch.sum(estimated * target, dim=-1, keepdim=True)
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        scaling_factor = dot_product / target_energy
        
        signal_component = scaling_factor * target
        noise_component = estimated - signal_component
        signal_energy = torch.sum(signal_component ** 2, dim=-1) + self.eps
        noise_energy = torch.sum(noise_component ** 2, dim=-1) + self.eps
        
        si_snr_db = 10 * torch.log10(signal_energy / noise_energy)
        
        # Return negative for loss minimization (we want to MAXIMIZE SI-SNR)
        return -si_snr_db  # Shape: (batch_size,)


class PIT_SI_SNR_Loss(nn.Module):
    """
    Permutation Invariant Training (PIT) with SI-SNR Loss.
    """
    
    def __init__(self, zero_mean: bool = True):
        super().__init__()
        self.si_snr_loss = SI_SNR_Loss(zero_mean=zero_mean)

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

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
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler  # ← ADDED: Store the scheduler
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
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        # ← ADDED: Save scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"  ✓ Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pth'):
        """Load training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            print(f"  ✗ Checkpoint not found: {filename}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # ← ADDED: Load scheduler state if it exists
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"  ✓ Checkpoint loaded: {filename} (epoch {self.epoch})")
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
        for epoch in range(start_epoch, num_epochs):  # ← Changed!
            self.epoch = epoch
            epoch_start_time = time.time()
            
            current_lr = self._get_current_lr()
            print(f"\nEpoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})")
            print("-" * 70)
            
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
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        
        # System parameters
        'device': 'cuda',
        'num_workers': 4,
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
    mixed_dir_train = "/gpfs/scratch/qp252467/datasets/source_separation/Libri2Mix_4s/train-clean-100_4s/mix_clean"
    source_dirs_train = ["/gpfs/scratch/qp252467/datasets/source_separation/Libri2Mix_4s/train-clean-100_4s/s1", \
                        "/gpfs/scratch/qp252467/datasets/source_separation/Libri2Mix_4s/train-clean-100_4s/s2"]  # List of source directories
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
    mixed_dir_val = '/gpfs/scratch/qp252467/datasets/source_separation/Libri2Mix_4s/dev_clean_4s/mix_clean'
    source_dirs_val = ['/gpfs/scratch/qp252467/datasets/source_separation/Libri2Mix_4s/dev_clean_4s/s1', \
                    '/gpfs/scratch/qp252467/datasets/source_separation/Libri2Mix_4s/dev_clean_4s/s2']  # List of source directories
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

    