import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def plot_multiclass_roc(y_true, y_score, title, out_path):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], label=f'micro-avg (AUC={roc_auc["micro"]:.3f})', linestyle=':', linewidth=3)
    colors = ['tab:blue', 'tab:orange', 'tab:green']; labels = ['Home', 'Draw', 'Away']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{labels[i]} (AUC={roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def savefig_tight(path, dpi=300, fig=None):
    """
    统一无告警地保存图像：
    - Matplotlib>=3.8: 使用 fig.set_layout_engine("tight")
    - 旧版本: 使用 bbox_inches='tight'
    """
    fig = fig or plt.gcf()
    set_engine = getattr(fig, "set_layout_engine", None)
    if callable(set_engine):
        # 新布局引擎方式，避免 UserWarning
        set_engine("tight")
        fig.savefig(path, dpi=dpi)
    else:
        # 兼容旧版本
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)