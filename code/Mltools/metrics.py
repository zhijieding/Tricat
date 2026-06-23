import numpy as np

def _row_norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

def calc_rps(y_true, y_prob, n_class=3):
    num = len(y_true)
    rps_sum = 0.0
    for i in range(num):
        pr = np.clip(y_prob[i], 1e-12, 1.0)
        pr = pr / pr.sum()
        oh = np.zeros(n_class, dtype=float)
        oh[int(y_true[i])] = 1.0
        cp = np.cumsum(pr); co = np.cumsum(oh)
        rps_sum += np.sum((cp - co) ** 2) / (n_class - 1)
    return rps_sum / max(num, 1)

def cdf3(P):
    P = _row_norm(P)
    return np.stack([P[:,0], P[:,0]+P[:,1]], axis=1)
