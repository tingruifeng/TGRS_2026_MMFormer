import numpy as np


def kappa(confusion_matrix):
    total = np.sum(confusion_matrix)
    if total == 0:
        return 0.0

    observed = np.trace(confusion_matrix) / total
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    expected = np.sum(row_sums * col_sums) / (total**2)
    return (observed - expected) / (1 - expected) if (1 - expected) != 0 else 0.0


def compute_metrics(confusion_matrix):
    class_totals = np.sum(confusion_matrix, axis=1)
    class_acc = np.divide(
        np.diag(confusion_matrix),
        class_totals,
        out=np.zeros_like(class_totals, dtype=float),
        where=class_totals != 0,
    )

    total = np.sum(confusion_matrix)
    oa = np.trace(confusion_matrix) / total if total != 0 else 0.0
    aa = float(np.mean(class_acc)) if class_acc.size else 0.0
    return {
        "CA": class_acc,
        "OA": float(oa),
        "AA": float(aa),
        "Kappa": float(kappa(confusion_matrix)),
    }


def summarize_history(history):
    ca = np.stack([item["CA"] for item in history], axis=1)
    oa = np.array([item["OA"] for item in history])
    aa = np.array([item["AA"] for item in history])
    kappa_values = np.array([item["Kappa"] for item in history])
    times = np.array([item.get("time", 0.0) for item in history])

    ddof = 1 if len(history) > 1 else 0
    return {
        "mean_CA": np.mean(ca, axis=1),
        "std_CA": np.std(ca, axis=1, ddof=ddof),
        "mean_OA": float(np.mean(oa)),
        "std_OA": float(np.std(oa, ddof=ddof)),
        "mean_AA": float(np.mean(aa)),
        "std_AA": float(np.std(aa, ddof=ddof)),
        "mean_Kappa": float(np.mean(kappa_values)),
        "std_Kappa": float(np.std(kappa_values, ddof=ddof)),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times, ddof=ddof)),
    }


def format_metric_report(metrics, class_names=None, total_time=None):
    lines = []
    class_acc = metrics["CA"]
    for idx, value in enumerate(class_acc):
        label = class_names[idx] if class_names else f"Class {idx + 1}"
        lines.append(f"{label} CA: {value:.4f}")
    lines.append(f"OA: {metrics['OA']:.4f}")
    lines.append(f"AA: {metrics['AA']:.4f}")
    lines.append(f"Kappa: {metrics['Kappa']:.4f}")
    if total_time is not None:
        lines.append(f"Time: {total_time:.2f} s")
    return "\n".join(lines)


def format_summary_report(summary, class_names=None):
    lines = []
    for idx, (mean_value, std_value) in enumerate(zip(summary["mean_CA"], summary["std_CA"])):
        label = class_names[idx] if class_names else f"Class {idx + 1}"
        lines.append(f"{label} CA: {mean_value:.4f} +/- {std_value:.4f}")
    lines.append(f"OA: {summary['mean_OA']:.4f} +/- {summary['std_OA']:.4f}")
    lines.append(f"AA: {summary['mean_AA']:.4f} +/- {summary['std_AA']:.4f}")
    lines.append(f"Kappa: {summary['mean_Kappa']:.4f} +/- {summary['std_Kappa']:.4f}")
    lines.append(f"Time: {summary['mean_time']:.2f} +/- {summary['std_time']:.2f} s")
    return "\n".join(lines)
