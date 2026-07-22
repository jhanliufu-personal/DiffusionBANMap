"""
Spike-triggered average (STA) axis tuning analysis.

Given a unit's responses to a set of stimuli and those stimuli's coordinates
in some feature space (e.g. PCA'd embeddings from a vision model), these
functions estimate the unit's preferred axis in that feature space and
quantify how selectively the unit is tuned to it.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def normalize_params_per_dim(params: np.ndarray) -> np.ndarray:
    """
    Scale each feature dimension to unit L2 norm across stimuli.

    Args:
        params: Stimulus feature matrix (n_stimuli, n_features)

    Returns:
        Normalized feature matrix, same shape
    """
    amp_dim = np.sqrt(np.sum(params**2, axis=0))
    amp_dim[amp_dim == 0] = 1
    return params / amp_dim[np.newaxis, :]


def compute_sta(
    resp: np.ndarray,
    params: np.ndarray,
    method: str = 'sta',
    normalize: bool = True,
    alpha: float = 1e-5
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Compute a unit's preferred axis from its responses and stimulus features.

    Args:
        resp: Response vector (n_stimuli,)
        params: Stimulus feature matrix (n_stimuli, n_features)
        method: 'sta', 'linear_regression', or 'ridge_regression'
        normalize: Whether to normalize feature dimensions first
        alpha: Ridge penalty (used only if method='ridge_regression')

    Returns:
        axis: The computed preferred axis (n_features,)
        ev: In-sample explained variance (only for 'linear_regression'; None otherwise)
    """
    if normalize:
        params = normalize_params_per_dim(params)

    nan_mask = ~np.isnan(resp)
    resp = resp[nan_mask]
    params = params[nan_mask, :]

    if method == 'linear_regression':
        X = np.column_stack([params, np.ones(params.shape[0])])
        coeffs, _, _, _ = np.linalg.lstsq(X, resp, rcond=None)
        axis = coeffs[:-1]

        ss_res = np.sum((resp - X @ coeffs) ** 2)
        ss_tot = np.sum((resp - np.mean(resp)) ** 2)
        ev = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    elif method == 'ridge_regression':
        resp_centered = resp - np.mean(resp)
        cov_stim = params.T @ params
        n_features = params.shape[1]
        identity_scaled = np.eye(n_features) * np.mean(np.diag(cov_stim)) * alpha

        axis = (resp_centered.T @ params) @ np.linalg.inv(cov_stim + identity_scaled)
        ev = None

    elif method == 'sta':
        resp_centered = resp - np.mean(resp)
        axis = resp_centered @ params
        ev = None

    else:
        raise ValueError(f"Unknown method: {method}")

    return axis, ev


def compute_explained_variance(obs_resp: np.ndarray, pred_resp: np.ndarray) -> float:
    """
    EV = 1 - sum((observed - predicted)^2) / sum((observed - mean)^2)
    """
    if len(obs_resp) != len(pred_resp):
        raise ValueError("Response vectors must have same length")

    ss_res = np.sum((obs_resp - pred_resp) ** 2)
    ss_tot = np.sum((obs_resp - np.mean(obs_resp)) ** 2)

    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def sta_cross_val_pref_only(
    resp_raw: np.ndarray,
    params: np.ndarray,
    method: str = 'sta',
    alpha: float = 1e-5
) -> Tuple[float, np.ndarray]:
    """
    Leave-one-out cross-validated explained variance along the preferred axis.

    For each held-out stimulus: recompute the axis on the remaining stimuli,
    project all stimuli onto it, fit a 1D affine map (response ~ projection)
    on the training stimuli, and predict the held-out response.

    Args:
        resp_raw: Response vector (n_stimuli,)
        params: Stimulus feature matrix (n_stimuli, n_features)
        method: 'sta', 'linear_regression', or 'ridge_regression'
        alpha: Ridge penalty

    Returns:
        ev: Cross-validated explained variance (clipped at -1)
        pred: Held-out predicted responses (n_stimuli,)
    """
    n_stim = len(resp_raw)
    params_norm = normalize_params_per_dim(params)

    pred = np.zeros(n_stim)

    for j in range(n_stim):
        ind_train = np.setdiff1d(np.arange(n_stim), [j])
        ind_test = j

        axis, _ = compute_sta(
            resp_raw[ind_train],
            params[ind_train, :],
            method=method,
            normalize=True,  # re-normalized per fold, matching STA_sub_cross_val_pref_only.m's
                              # call into analysis_STA with toNorm defaulting to true -- the
                              # per-dim norm is computed from the training subset only, distinct
                              # from params_norm above (which uses the full stimulus set and is
                              # only used for projecting, not for fitting the axis itself)
            alpha=alpha
        )

        prj = params_norm @ axis

        coeffs = np.polyfit(prj[ind_train], resp_raw[ind_train], 1)
        pred[ind_test] = np.polyval(coeffs, prj[ind_test])

    ss_res = np.sum((pred - resp_raw) ** 2)
    ss_tot = np.sum((resp_raw - np.mean(resp_raw)) ** 2)

    ev = 1 - (ss_res / ss_tot) if ss_tot > 0 else -1
    return max(ev, -1), pred


def compute_orthogonal_variance(
    resp_raw: np.ndarray,
    params: np.ndarray,
    sta: np.ndarray,
    n_components_pca: Optional[int] = None
) -> Dict:
    """
    Compute EV/selectivity relative to a principal orthogonal axis.

    1. Subtract the preferred axis's component from every stimulus vector.
    2. PCA the residuals; take the top PC as the "orthogonal" control axis.
    3. Compare EV along the preferred vs. orthogonal axis.

    Args:
        resp_raw: Response vector (n_stimuli,)
        params: Stimulus feature matrix (n_stimuli, n_features)
        sta: Preferred axis (n_features,)
        n_components_pca: PCA components to fit when finding the orthogonal
            axis (only the first is used). Defaults to min(n_features-1, n_stimuli).

    Returns:
        Dict with 'preferred_ev', 'orthogonal_ev', 'selectivity_index',
        'orth_axis', 'variance_ratio', 'proj_pref', 'proj_orth'
    """
    from sklearn.decomposition import PCA

    params_norm = normalize_params_per_dim(params)
    n_features = params_norm.shape[1]
    n_stimuli = params_norm.shape[0]

    if n_components_pca is None:
        n_components_pca = min(n_features - 1, n_stimuli)

    # Preferred axis -- project onto the unit-normalized sta, matching
    # STA_figure_clean.m's `value_sta_prj = (sta/norm(sta))*para'`
    sta_normalized = sta / np.linalg.norm(sta)
    proj_pref = params_norm @ sta_normalized
    coeffs_pref = np.polyfit(proj_pref, resp_raw, 1)
    pred_pref = np.polyval(coeffs_pref, proj_pref)
    ev_pref = compute_explained_variance(resp_raw, pred_pref)

    # Principal orthogonal axis
    params_sub_sta = params_norm - np.outer(proj_pref, sta_normalized)

    n_components_to_fit = min(n_components_pca, n_stimuli, n_features)
    pca_orth = PCA(n_components=n_components_to_fit)
    pca_orth.fit(params_sub_sta)
    orth_axis = pca_orth.components_[0]

    proj_orth = params_sub_sta @ orth_axis
    coeffs_orth = np.polyfit(proj_orth, resp_raw, 1)
    pred_orth = np.polyval(coeffs_orth, proj_orth)
    ev_orth = compute_explained_variance(resp_raw, pred_orth)

    if (ev_pref + ev_orth) > 0:
        selectivity_index = (ev_pref - ev_orth) / (ev_pref + ev_orth)
    else:
        selectivity_index = 0.0

    var_pref = np.var(proj_pref)
    var_orth = np.var(proj_orth)
    variance_ratio = var_orth / var_pref if var_pref > 0 else 0.0

    return {
        'preferred_ev': ev_pref,
        'orthogonal_ev': ev_orth,
        'selectivity_index': selectivity_index,
        'orth_axis': orth_axis,
        'variance_ratio': variance_ratio,
        'proj_pref': proj_pref,
        'proj_orth': proj_orth,
    }


def compute_axis_orthogonality(axes: np.ndarray) -> Dict:
    """
    Pairwise orthogonality of a population's preferred axes.

    Args:
        axes: Preferred axes for each unit (n_units, n_features)

    Returns:
        Dict with 'mean_abs_cosine_similarity', 'median_abs_cosine_similarity',
        'cosine_similarity_matrix', 'mean_pairwise_angle_deg'
    """
    n_units = axes.shape[0]
    axes_norm = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-10)
    cosine_sim_matrix = axes_norm @ axes_norm.T

    upper_tri = np.triu_indices(n_units, k=1)
    pairwise_cosines = np.clip(cosine_sim_matrix[upper_tri], -1.0, 1.0)
    abs_cosines = np.abs(pairwise_cosines)

    angles_deg = np.degrees(np.arccos(abs_cosines))

    return {
        'mean_abs_cosine_similarity': float(np.mean(abs_cosines)),
        'median_abs_cosine_similarity': float(np.median(abs_cosines)),
        'cosine_similarity_matrix': cosine_sim_matrix,
        'mean_pairwise_angle_deg': float(np.mean(angles_deg)),
    }


def compute_population_axis_tuning(
    responses: np.ndarray,
    params: np.ndarray,
    method: str = 'sta',
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Run axis tuning analysis for every unit (neuron or latent) in a population.

    Args:
        responses: Response matrix (n_units, n_stimuli)
        params: Stimulus feature matrix (n_stimuli, n_features), e.g. PCA'd
            embeddings. Assumed already reduced to the desired dimensionality.
        method: 'sta', 'linear_regression', or 'ridge_regression'
        show_progress: Show a tqdm progress bar

    Returns:
        Dict with per-unit arrays: 'explained_variance', 'predictions', 'axes',
        'orthogonal_axes', 'orthogonal_ev', 'selectivity_index', 'variance_ratio',
        plus population-level 'mean_abs_cosine_similarity',
        'median_abs_cosine_similarity', 'cosine_similarity_matrix',
        'mean_pairwise_angle_deg'
    """
    n_units, n_stimuli = responses.shape
    n_features = params.shape[1]

    explained_variance = np.zeros(n_units)
    predictions = np.zeros((n_units, n_stimuli))
    axes = np.zeros((n_units, n_features))
    orthogonal_axes = np.zeros((n_units, n_features))
    orthogonal_ev = np.zeros(n_units)
    selectivity_index = np.zeros(n_units)
    variance_ratio = np.zeros(n_units)

    iterator = range(n_units)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"Axis tuning ({method})")

    for unit_idx in iterator:
        resp = responses[unit_idx, :]

        ev, pred = sta_cross_val_pref_only(resp, params, method=method)
        explained_variance[unit_idx] = ev
        predictions[unit_idx, :] = pred

        axis, _ = compute_sta(resp, params, method=method, normalize=True)
        axes[unit_idx, :] = axis

        orth_metrics = compute_orthogonal_variance(resp, params, axis)
        orthogonal_axes[unit_idx, :] = orth_metrics['orth_axis']
        orthogonal_ev[unit_idx] = orth_metrics['orthogonal_ev']
        selectivity_index[unit_idx] = orth_metrics['selectivity_index']
        variance_ratio[unit_idx] = orth_metrics['variance_ratio']

    orthogonality_metrics = compute_axis_orthogonality(axes)

    return {
        'explained_variance': explained_variance,
        'predictions': predictions,
        'axes': axes,
        'orthogonal_axes': orthogonal_axes,
        'orthogonal_ev': orthogonal_ev,
        'selectivity_index': selectivity_index,
        'variance_ratio': variance_ratio,
        **orthogonality_metrics,
    }
