#!/usr/bin/env python3

import sys
import os
import cv2
import torch
from torchvision.transforms import v2
from utils import get_model
import numpy as np
from typing import Tuple, Optional


def plot_img_and_bbox(
    img: np.ndarray,
    boxes: np.ndarray,
    out_file: str,
    rgb: Optional[Tuple[int, int, int]] = None,
) -> None:
    if rgb is None:
        rgb = (0, 0, 255)
    for x1, y1, x2, y2 in boxes:
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    cv2.imwrite(out_file, img)


def predict(file_path: str):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)])
    batch = transform(img).unsqueeze(0)

    model = get_model("bee_count_fasterrcnn_resnet50_fpn.pt")
    model.eval()
    outputs = model(batch)
    boxes = outputs[0]["boxes"].detach().numpy().astype(np.int32)

    output_file_name = "predicted.jpg"
    plot_img_and_bbox(img, boxes, output_file_name, (0, 0, 255))

    print(f"Output file: {output_file_name}")
    print(f"Total bees: {len(outputs[0]['scores'])}")


def main() -> None:
    try:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Error: '{file_path}' not found")
            sys.exit(1)
    except IndexError:
        print("Usage: ./sample_solution.py <image file>")
        sys.exit(1)

    predict(file_path)


if __name__:
    main()
