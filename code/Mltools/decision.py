import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def classwise_temp(P, aH=1.0, aD=1.0, aA=1.0):
    Z = np.log(np.clip(P,1e-12,1-1e-12))
    Z = np.column_stack([Z[:,0]*aH, Z[:,1]*aD, Z[:,2]*aA])
    Q = np.exp(Z); return Q / np.clip(Q.sum(axis=1, keepdims=True), 1e-12, None)

def predict_with_params(P_dec, tau_H, tau_D, tau_A, gamma, m_draw=0.05):
    S = np.column_stack([P_dec[:,0]/tau_H, P_dec[:,1]/tau_D, P_dec[:,2]/tau_A])
    S_max = np.maximum(S[:, 0], S[:, 2])
    is_draw = (S[:, 1] + m_draw >= S_max) & (P_dec[:, 1] >= tau_D)
    pred = np.full(len(P_dec), -1, dtype=int)
    pred[is_draw] = 1
    idx = ~is_draw
    Ph, Pa = P_dec[idx,0], P_dec[idx,2]
    denom = np.clip(Ph + Pa, 1e-12, None)
    p_home_cond = Ph / denom
    pred_nd = np.where(p_home_cond >= 0.5 + gamma, 0,
                np.where(p_home_cond <= 0.5 - gamma, 2,
                    np.where(S[idx,0] > S[idx,2], 0, 2)))
    pred[idx] = pred_nd
    return pred

def evaluate_decision(y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2], zero_division=0
    )
    return float(f1.mean()), prec, rec, f1

def search_decision_params_strong(P_cal, y_cal):
    rate_H = float((y_cal == 0).mean())
    rate_D = float((y_cal == 1).mean())
    rate_A = float((y_cal == 2).mean())

    alpha_sets = [
        (1.0, 1.0, 1.0),
        (1.1, 1.3, 1.1),
        (1.2, 1.4, 1.2),
        (1.3, 1.5, 1.3),
        (1.0, 1.6, 1.2),
        (1.2, 1.2, 1.4),
        (1.3, 1.4, 1.4),
        (1.1, 1.5, 1.3),
    ]
    grid_H = np.linspace(0.35, 0.65, 7)
    grid_D = np.linspace(0.18, 0.52, 9)
    grid_A = np.linspace(0.30, 0.70, 7)
    gammas = np.linspace(0.00, 0.24, 13)

    floors = [0.95, 0.80, 0.70]
    prec_floor = 0.25

    best = None
    for floor in floors:
        best = (-1.0, (1.0,1.0,1.0), 0.5, 0.35, 0.5, 0.10)
        for (aH, aD, aA) in alpha_sets:
            P_dec = classwise_temp(P_cal, aH, aD, aA)
            for th in grid_H:
                for td in grid_D:
                    for ta in grid_A:
                        for g in gammas:
                            y_pred = predict_with_params(P_dec, th, td, ta, g)
                            ph = (y_pred == 0).mean(); pd = (y_pred == 1).mean(); pa = (y_pred == 2).mean()
                            if (ph < floor*rate_H) or (pd < floor*rate_D) or (pa < floor*rate_A):
                                continue
                            _, prec, _, _ = precision_recall_fscore_support(
                                y_cal, y_pred, labels=[0,1,2], zero_division=0
                            )
                            if (prec[0] < prec_floor) or (prec[1] < prec_floor) or (prec[2] < prec_floor):
                                continue
                            score, _, _, _ = evaluate_decision(y_cal, y_pred)
                            if score > best[0]:
                                best = (score, (aH,aD,aA), float(th), float(td), float(ta), float(g))
        if best[0] >= 0:
            break

    if best[0] < 0:
        for (aH, aD, aA) in alpha_sets:
            P_dec = classwise_temp(P_cal, aH, aD, aA)
            for th in grid_H:
                for td in grid_D:
                    for ta in grid_A:
                        for g in gammas:
                            y_pred = predict_with_params(P_dec, th, td, ta, g)
                            score, _, _, _ = evaluate_decision(y_cal, y_pred)
                            if (best is None) or (score > best[0]):
                                best = (score, (aH,aD,aA), float(th), float(td), float(ta), float(g))

    score, (aH,aD,aA), tau_H, tau_D, tau_A, gamma = best
    print(f"[决策参数] α_H={aH:.2f}, α_D={aD:.2f}, α_A={aA:.2f} | τ_H={tau_H:.2f}, τ_D={tau_D:.2f}, τ_A={tau_A:.2f} | γ={gamma:.2f} | 校准集宏F1={score:.4f}")
    return (aH,aD,aA), tau_H, tau_D, tau_A, gamma
