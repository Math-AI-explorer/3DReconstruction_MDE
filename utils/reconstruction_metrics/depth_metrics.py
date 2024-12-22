import numpy as np


def absrel_depth_error(gts: np.ndarray, preds: np.ndarray) -> float:
    assert all([
        len(gts.shape) == 3,
        len(preds.shape) == 3,
        gts.shape == preds.shape,
        (gts >= 0).all(), (preds >= 0).all(),
    ])
    mask = gts > 1e-8
    return np.mean(np.abs(gts[mask] - preds[mask]) / gts[mask])

def acc_under_threshold_depth_error(
    gts: np.ndarray, preds: np.ndarray, threshold: float=1.25
) -> float:
    assert all([
        len(gts.shape) == 3,
        len(preds.shape) == 3,
        gts.shape == preds.shape,
        (gts >= 0).all(), (preds >= 0).all(),
    ])
    eps = 1e-8
    mask = np.logical_and(gts > eps, preds > eps)
    preds_pos, gts_pos = preds[mask], gts[mask]

    return np.mean(np.maximum(gts_pos/preds_pos, preds_pos/gts_pos) < threshold)

def rms_depth_error(gts: np.ndarray, preds: np.ndarray) -> float:
    assert all([
        len(gts.shape) == 3,
        len(preds.shape) == 3,
        gts.shape == preds.shape,
        (gts >= 0).all(), (preds >= 0).all(),
    ])

    return np.sqrt(np.mean(np.power(gts - preds, 2)))

def rms_log_depth_error(gts: np.ndarray, preds: np.ndarray) -> float:
    assert all([
        len(gts.shape) == 3,
        len(preds.shape) == 3,
        gts.shape == preds.shape,
        (gts >= 0).all(), (preds >= 0).all(),
    ])
    eps = 1e-8
    mask = np.logical_and(gts > eps, preds > eps)
    preds_pos, gts_pos = preds[mask], gts[mask]

    return np.sqrt(np.mean(np.power(np.log(gts_pos) - np.log(preds_pos), 2)))