import cv2
import os
import unittest

import torch
from torchvision.transforms import v2

from utils import get_model


class TestBeeCount(unittest.TestCase):
    def setUp(self):
        test_root = os.path.dirname(__file__)
        test_img_path = os.path.join(test_root, "test_img.jpg")
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)])
        img = cv2.imread(test_img_path, cv2.IMREAD_COLOR_RGB)
        
        self.batch = transform(img).unsqueeze(0)
        self.model_path = os.path.join(test_root, "../bee_count_fasterrcnn_resnet50_fpn.pt")

    def test_bee_count(self):
        model = get_model(self.model_path)
        model.eval()
        outputs = model(self.batch)
        self.assertEqual(len(outputs[0]["scores"]), 48)
