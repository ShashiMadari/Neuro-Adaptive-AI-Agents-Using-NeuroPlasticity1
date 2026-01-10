import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class ObjectManipulationAnalyzer:
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect objects and their manipulation states"""
        results = self.model(image_path)
        detections = []
        
        for result in results:
            for box in result.boxes:
                obj_info = {
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                    "center": self._get_center(box.xyxy[0]),
                    "manipulation_state": self._assess_manipulation(box, result)
                }
                detections.append(obj_info)
        
        return detections
    
    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def _assess_manipulation(self, box, result):
        """Assess if object is being manipulated"""
        bbox = box.xyxy[0].tolist()
        center_x, center_y = self._get_center(bbox)
        
        # Simple heuristic: if object is in lower center, likely being held
        image_height = result.orig_shape[0]
        if center_y > image_height * 0.6:  # Lower portion of image
            return "possibly_held"
        return "stationary"

# Singleton instance
object_analyzer = None

def get_object_analyzer():
    global object_analyzer
    if object_analyzer is None:
        object_analyzer = ObjectManipulationAnalyzer()
    return object_analyzer