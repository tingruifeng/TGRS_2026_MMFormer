from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


COLOR_MAPS = {
    "HC": {
        0: [0, 0, 0],
        1: [140, 67, 46],
        2: [0, 0, 255],
        3: [255, 100, 0],
        4: [0, 255, 200],
        5: [164, 75, 155],
        6: [101, 174, 255],
    },
    "NF": {
        0: [0, 0, 0],
        1: [230, 57, 70],
        2: [29, 155, 240],
        3: [0, 180, 166],
        4: [255, 163, 0],
        5: [168, 85, 247],
        6: [220, 38, 38],
        7: [245, 158, 11],
        8: [107, 114, 128],
        9: [5, 150, 105],
    },
    "UP": {
        0: [0, 0, 0],
        1: [251, 113, 133],
        2: [59, 130, 246],
        3: [34, 197, 94],
        4: [249, 115, 22],
        5: [139, 92, 246],
        6: [236, 72, 153],
        7: [163, 163, 163],
        8: [20, 184, 166],
        9: [147, 51, 234],
    },
}


def labels_to_rgb(labels, dataset_name):
    color_map = COLOR_MAPS[dataset_name.upper()]
    flat_labels = labels.reshape(-1)
    rgb = np.zeros((flat_labels.shape[0], 3), dtype=float)
    for idx, item in enumerate(flat_labels):
        rgb[idx] = np.array(color_map.get(int(item), [0, 0, 0]), dtype=float) / 255.0
    return rgb.reshape(labels.shape[0], labels.shape[1], 3)


def predict_all(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data, _ in tqdm(loader, desc="Predicting map"):
            logits = model(batch_data.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.append(pred)
    return np.concatenate(predictions)


def predictions_to_label_map(predictions, ground_truth):
    height, width = ground_truth.shape
    label_map = np.zeros((height, width), dtype=np.int64)
    pred_idx = 0
    for row in range(height):
        for col in range(width):
            if int(ground_truth[row, col]) == 0:
                continue
            label_map[row, col] = int(predictions[pred_idx]) + 1
            pred_idx += 1
    return label_map


def save_rgb_map(rgb_map, ground_truth, save_path, dpi=300):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(rgb_map)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return save_path


def save_prediction_and_gt_maps(model, loader, device, ground_truth, dataset_name, save_dir, oa):
    save_dir = Path(save_dir)
    map_dir = save_dir / "classification_maps"
    predictions = predict_all(model, loader, device)
    pred_label_map = predictions_to_label_map(predictions, ground_truth)

    pred_rgb = labels_to_rgb(pred_label_map, dataset_name)
    gt_rgb = labels_to_rgb(ground_truth, dataset_name)

    pred_path = save_rgb_map(pred_rgb, ground_truth, map_dir / f"prediction_map_OA_{oa:.4f}.png")
    gt_path = save_rgb_map(gt_rgb, ground_truth, map_dir / "ground_truth_map.png")
    np.save(map_dir / "best_prediction_results.npy", predictions)
    return pred_path, gt_path
