import numpy as np
from sklearn.linear_model import LogisticRegression

def _logit_clip(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p/(1-p))

def _sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def platt_fit(raw_p, y_binary, C=1e6):
    z = _logit_clip(raw_p).reshape(-1, 1)
    lr = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, class_weight='balanced')
    lr.fit(z, y_binary.astype(int))
    return lr

def platt_apply(lr, raw_p):
    z = _logit_clip(raw_p).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]
