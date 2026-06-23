import numpy as np

def odds_to_probs(arr3):
    arr = arr3.astype(float)
    p = 1.0 / np.clip(arr, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p

def build_soft_dataset(X, y_hard, soft_pos, lam=0.7):
    y_smooth = (1 - lam) * y_hard + lam * soft_pos
    X_dup = np.repeat(X, 2, axis=0)
    y_dup = np.empty(2 * len(y_hard), dtype=int)
    w_dup = np.empty(2 * len(y_hard), dtype=float)
    y_dup[0::2] = 1; w_dup[0::2] = np.clip(y_smooth, 1e-6, 1-1e-6)
    y_dup[1::2] = 0; w_dup[1::2] = np.clip(1 - y_smooth, 1e-6, 1-1e-6)
    return X_dup, y_dup, w_dup
