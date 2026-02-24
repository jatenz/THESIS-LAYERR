import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1
    return cm

def accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    correct = 0
    for a, b in zip(y_true, y_pred):
        if a == b:
            correct += 1
    return float(correct / len(y_true))

def f1_macro(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels)
    n = len(labels)
    f1s = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), cm

def thresholded_accuracy(y_true, y_pred, y_conf, threshold):
    if len(y_true) == 0:
        return 0.0, 0.0, 0
    keep = [i for i, c in enumerate(y_conf) if float(c) >= float(threshold)]
    if len(keep) == 0:
        return 0.0, 0.0, 0
    yt = [y_true[i] for i in keep]
    yp = [y_pred[i] for i in keep]
    acc = accuracy(yt, yp)
    coverage = float(len(keep) / len(y_true))
    return float(acc), float(coverage), int(len(keep))