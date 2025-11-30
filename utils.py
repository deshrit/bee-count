import os
import torch
import torchvision

from zipfile import ZipFile
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from huggingface_hub import snapshot_download


def extract_zip(zip_file: str, dir: str = ".") -> None:
    if not os.path.exists(zip_file):
        print(f"Error: ZIP file not found at '{zip_file}'")
        return

    os.makedirs(dir, exist_ok=True)

    try:
        with ZipFile(zip_file, "r") as zip_object:
            zip_object.extractall(dir)
        print(f"Successfully extracted '{zip_file}' to '{dir}'")
    except Exception as e:
        print(f"An error occurred during extraction: {e}")


def get_model_cls(num_classes=2) -> FasterRCNN:
    """Helper to get FasterRCNN model class"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def download_model_hugging_face(model_repo_id="deshrit/bee-count", dir=".") -> None:
    try:
        print("## Downloading trained model from hugging face ##\n")
        snapshot_download(repo_id=model_repo_id, local_dir=dir)
    except Exception as e:
        print(f"Error downloading model from hugging face: {e}")


def get_model(model_path: str) -> FasterRCNN:
    """Helper to load trained FasterRCNN model from state dict"""

    if not os.path.exists(model_path):
        download_model_hugging_face()

    model_loaded = get_model_cls()
    state_dict = torch.load(model_path)
    model_loaded.load_state_dict(state_dict)
    return model_loaded
