# ════════════════════════════════════════════════════════════════════════════════
# SVD Evolution Analysis
# ════════════════════════════════════════════════════════════════════════════════

import torch
import numpy as np
from typing import Dict, List, Tuple

def compute_svd(weight: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Break down a weight matrix using SVD.
    
    Args:
        weight: A 2D tensor (matrix)
        
    Returns:
        Dictionary with U, S, Vt tensors
    """
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
    
    return {
        'U': U,
        'S': S,
        'Vt': Vt,
    }

def compute_effective_rank(S: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute the entropy-based effective rank from a vector of singular values.

    Args:
        S:   1D tensor of singular values (non-negative, descending)
        eps: Small constant for numerical stability

    Returns:
        Scalar tensor: effective rank = exp(H(p)), where p_i = sigma_i / sum(sigma)
    """
    # Normalize to get a probability distribution over singular values
    S_norm = S / (S.sum() + eps)

    # Keep only entries that are numerically meaningful
    S_norm = S_norm[S_norm > eps]

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -torch.sum(S_norm * torch.log(S_norm))

    # Effective rank = exp(H)
    return torch.exp(entropy)

def _interpolate_sv_at_rank(S: torch.Tensor, effective_rank: torch.Tensor) -> torch.Tensor:
    """
    Linearly interpolate the singular value spectrum at a fractional rank index.

    Treats effective_rank as a 1-based fractional index into S, e.g.:
        effective_rank = 64.38  →  S[63] + 0.38 * (S[64] - S[63])

    Args:
        S:              Full singular value vector, shape (r,), descending order.
        effective_rank: Scalar tensor, the computed effective rank.

    Returns:
        Scalar tensor: the interpolated singular value.
        Returns torch.tensor(float('nan')) if the rank index is out of bounds.
    """
    r    = S.shape[0]
    er   = effective_rank.item()       # Python float, e.g. 64.38

    lo   = int(np.floor(er)) - 1      # 0-based lower index  (63)
    hi   = lo + 1                     # 0-based upper index  (64)
    frac = er - int(np.floor(er))     # fractional part       (0.38)

    # Guard: both neighbours must exist in the full spectrum
    if lo < 0 or hi >= r:
        return torch.tensor(float('nan'))

    sigma_lo = S[lo]
    sigma_hi = S[hi]
    return sigma_lo + frac * (sigma_hi - sigma_lo)

def compute_condition_number(S: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute the condition number as the ratio of the largest to the smallest singular value.

    Args:
        S:   1D tensor of singular values (descending)
        eps: Threshold below which the smallest singular value is considered zero

    Returns:
        Scalar tensor: condition number (torch.inf if smallest SV ~ 0)
    """
    sigma_min = S[-1]
    if sigma_min > eps:
        return S[0] / sigma_min
    else:
        return torch.tensor(torch.inf, dtype=S.dtype, device=S.device)

def _compute_individual_angles(
    V_curr: torch.Tensor,   # (n, m)  column-orthonormal
    V_next: torch.Tensor,   # (n, m)  column-orthonormal
    k:      int,
) -> np.ndarray:
    """
    Angle in degrees between matching column vectors across two epochs.

    theta_i = arccos(|v_i_curr . v_i_next|)   for i = 0 .. k-1

    The absolute dot product resolves SVD sign ambiguity:
    v and -v span the same 1-D subspace, so we always measure the acute angle between them.

    Returns:
        angles: np.ndarray, shape (k,), values in [0, 90] degrees.
    """
    angles = np.empty(k, dtype=np.float32)

    for i in range(k):
        v_c = V_curr[:, i]
        v_n = V_next[:, i]

        # Re-normalise defensively (should already be unit vectors)
        v_c = v_c / (v_c.norm() + 1e-12)
        v_n = v_n / (v_n.norm() + 1e-12)

        cos_sim   = torch.clamp(torch.abs(torch.dot(v_c, v_n)), 0.0, 1.0)
        angles[i] = (torch.acos(cos_sim) * (180.0 / torch.pi)).item()

    return angles

def _compute_principal_angles(
    V_curr: torch.Tensor,   # (n, k)  column-orthonormal  (basis of subspace A)
    V_next: torch.Tensor,   # (n, k)  column-orthonormal  (basis of subspace B)
) -> np.ndarray:
    """
    Principal angles between two k-dimensional subspaces via SVD.

    Algorithm (Bjorck & Golub 1973):
        1. Form the cross-Gram matrix  M = V_curr^T @ V_next   (k x k)
        2. SVD: M = P diag(sigma) Q^T
        3. Principal angles: theta_i = arccos(clamp(sigma_i, 0, 1))

    The singular values of M are the cosines of the principal angles.
    They are naturally in [0, 1] (up to floating-point noise), so clamping is just a safety measure.

    Returns:
        principal_angles: np.ndarray shape (k,), ascending order, degrees.Range [0, 90] degrees.
    """
    # Cross-Gram matrix  (k x k)
    M = V_curr.T @ V_next                    # torch.Tensor (k, k)

    # Singular values of M = cosines of principal angles
    # torch.linalg.svdvals returns values in descending order
    sigma = torch.linalg.svdvals(M)          # shape (k,), descending

    # Clamp to [0, 1] against floating-point overshoot
    sigma = torch.clamp(sigma, 0.0, 1.0)

    # Convert to angles and sort ascending (smallest angle first)
    angles_deg = (torch.acos(sigma) * (180.0 / torch.pi)).cpu().numpy()
    angles_deg = np.sort(angles_deg)         # ascending: shape (k,)

    return angles_deg.astype(np.float32)

def _compute_grassmann_distance(principal_angles_deg: np.ndarray) -> float:
    """
    Geodesic distance on the Grassmannian Gr(k, n).

    d_G(A, B) = || theta ||_2   where theta are the principal angles in RADIANS.

    This is the standard Riemannian (geodesic) distance on the Grassmannian.
    It equals zero iff the two subspaces are identical, and reaches its
    maximum of sqrt(k) * pi/2 when the subspaces are fully orthogonal.

    Args:
        principal_angles_deg: np.ndarray shape (k,), values in degrees.

    Returns:
        Scalar float, distance in radians.
    """
    angles_rad = principal_angles_deg * (np.pi / 180.0)
    return float(np.sqrt(np.sum(angles_rad ** 2)))

def analyze_weight_evolution(
    epochs:              List[int],
    linear_weights_list: List[Dict[str, torch.Tensor]],
    top_k:               int = 10,
) -> Dict[str, Dict]:
    """
    Track how each linear weight matrix's SVD changes across training epochs.

    All per-epoch quantities are indexed by evolution['epochs'].
    All per-transition quantities are indexed by evolution['epochs'][1:],
    i.e. the epoch at which the NEW state was observed.

    Args:
        epochs:               Ordered list of epoch numbers.
        linear_weights_list:  One state-dict (filtered to linear weights)
                              per epoch, aligned with `epochs`.
        top_k:                How many top singular values/vectors to track.
                              Also controls the number of individual angles
                              and principal angles stored.

    Returns:
        Dict keyed by weight name.  Each value is a Dict with:

        Per-epoch arrays  (length = n_epochs)
        ─────────────────────────────────────
        'name'               str
        'epochs'             List[int]
        'shapes'             List[Tuple[int, int]]
        'singular_values'    List[Tensor (k,)]     top-k singular values
        'left_vectors'       List[Tensor (m, k)]   top-k left singular vectors
        'right_vectors'      List[Tensor (n, k)]   top-k right singular vectors
        'effective_ranks'    List[Tensor scalar]
        'condition_numbers'  List[Tensor scalar]
        'effective_rank_sv'  List[Tensor scalar]   σ at fractional eff-rank idx

        Per-transition arrays  (length = n_epochs - 1, x-axis = epochs[1:])
        ─────────────────────────────────────────────────────────────────────
        'left_angles'              ndarray (n-1, k)   individual vector angles
        'right_angles'             ndarray (n-1, k)   individual vector angles
        'left_principal_angles'    ndarray (n-1, k)   subspace principal angles
        'right_principal_angles'   ndarray (n-1, k)   subspace principal angles
        'left_grassmann_dist'      ndarray (n-1,)     geodesic dist on Gr(k,m)
        'right_grassmann_dist'     ndarray (n-1,)     geodesic dist on Gr(k,n)
    """
    if len(linear_weights_list) == 0:
        raise ValueError("No linear weights found.")

    evolutions:  Dict[str, Dict] = {}
    weight_names = list(linear_weights_list[0].keys())

    print(f"Analyzing SVD evolution for {len(weight_names)} weight matrices...")

    for name in weight_names:

        # ------------------------------------------------------------------ #
        # Per-epoch storage
        # ------------------------------------------------------------------ #
        evolution: Dict = {
            'name':              name,
            'epochs':            [],
            'shapes':            [],
            'singular_values':   [],
            'left_vectors':      [],
            'right_vectors':     [],
            'effective_ranks':   [],
            'condition_numbers': [],
            'effective_rank_sv': [],
        }

        # ------------------------------------------------------------------ #
        # Pass 1 — per-epoch SVD quantities
        # ------------------------------------------------------------------ #
        for epoch, weights in zip(epochs, linear_weights_list):

            if name not in weights:
                print(f"  Warning: '{name}' not found at epoch {epoch} — skipping.")
                continue

            W = weights[name]
            if W.shape[0] < 2 or W.shape[1] < 2:
                continue

            svd = compute_svd(W)
            S, U, Vt = svd['S'], svd['U'], svd['Vt']

            # Derived metrics on the FULL spectrum (before truncation)
            eff_rank    = compute_effective_rank(S)
            cond_num    = compute_condition_number(S)
            eff_rank_sv = _interpolate_sv_at_rank(S, eff_rank)

            k = min(top_k, S.shape[0])

            evolution['epochs'].append(epoch)
            evolution['shapes'].append(tuple(W.shape))
            evolution['singular_values'].append(S[:k])
            evolution['left_vectors'].append(U[:, :k])          # (m, k)
            evolution['right_vectors'].append(Vt[:k, :].T)      # (n, k)
            evolution['effective_ranks'].append(eff_rank)
            evolution['condition_numbers'].append(cond_num)
            evolution['effective_rank_sv'].append(eff_rank_sv)

        # Skip weights with insufficient data
        n = len(evolution['epochs'])
        if n == 0:
            continue

        # ------------------------------------------------------------------ #
        # Pass 2 — per-transition geometry (n-1 values each)
        #
        # x-axis for all transition quantities: evolution['epochs'][1:]
        # ------------------------------------------------------------------ #
        k_stored = evolution['singular_values'][0].shape[0]  # actual k after clamp

        # Pre-allocate all transition arrays
        left_angles           = np.full((n - 1, k_stored), np.nan, dtype=np.float32)
        right_angles          = np.full((n - 1, k_stored), np.nan, dtype=np.float32)
        left_principal_angles  = np.full((n - 1, k_stored), np.nan, dtype=np.float32)
        right_principal_angles = np.full((n - 1, k_stored), np.nan, dtype=np.float32)
        left_grassmann_dist   = np.full((n - 1,),           np.nan, dtype=np.float32)
        right_grassmann_dist  = np.full((n - 1,),           np.nan, dtype=np.float32)

        for t in range(n - 1):
            U_curr = evolution['left_vectors'][t]       # (m, k)
            U_next = evolution['left_vectors'][t + 1]   # (m, k)
            V_curr = evolution['right_vectors'][t]      # (n, k)
            V_next = evolution['right_vectors'][t + 1]  # (n, k)

            # ── Individual vector angles ──────────────────────────────── #
            left_angles[t]  = _compute_individual_angles(U_curr, U_next, k_stored)
            right_angles[t] = _compute_individual_angles(V_curr, V_next, k_stored)

            # ── Principal angles between subspaces ────────────────────── #
            pa_left  = _compute_principal_angles(U_curr, U_next)  # (k,) degrees
            pa_right = _compute_principal_angles(V_curr, V_next)  # (k,) degrees

            left_principal_angles[t]  = pa_left
            right_principal_angles[t] = pa_right

            # ── Grassmann geodesic distance ───────────────────────────── #
            left_grassmann_dist[t]  = _compute_grassmann_distance(pa_left)
            right_grassmann_dist[t] = _compute_grassmann_distance(pa_right)

        evolution['left_angles']             = left_angles
        evolution['right_angles']            = right_angles
        evolution['left_principal_angles']   = left_principal_angles
        evolution['right_principal_angles']  = right_principal_angles
        evolution['left_grassmann_dist']     = left_grassmann_dist
        evolution['right_grassmann_dist']    = right_grassmann_dist

        evolutions[name] = evolution
        print(
            f"  {name}: {n} epochs  |  "
            f"shape {evolution['shapes'][0]}  |  "
            f"k={k_stored}  |  "
            f"{n - 1} transitions computed"
        )

    print(f"\nTotal weight matrices analyzed: {len(evolutions)}")
    return evolutions
    
