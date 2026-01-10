from typing import Dict, Any, List
from vision.object_analyzer import get_object_analyzer

class ExtendedVisionReasoner:
    def __init__(self):
        self.object_analyzer = get_object_analyzer()
    
    def analyze_image_comprehension(self, image_path: str, query: str = None) -> Dict[str, Any]:
        """Comprehensive image understanding with object manipulation"""
        analysis = {
            "objects_detected": [],
            "spatial_relationships": [],
            "manipulation_analysis": {},
            "scene_context": "",
            "action_inference": ""
        }
        
        # Detect objects
        objects = self.object_analyzer.detect_objects(image_path)
        analysis["objects_detected"] = objects
        
        # Analyze spatial relationships
        analysis["spatial_relationships"] = self._analyze_spatial_relationships(objects)
        
        # Infer manipulation states
        analysis["manipulation_analysis"] = self._analyze_manipulation(objects)
        
        # Infer scene context and actions
        analysis["scene_context"] = self._infer_scene_context(objects)
        analysis["action_inference"] = self._infer_actions(objects, query)
        
        return analysis
    
    def _analyze_spatial_relationships(self, objects: List[Dict[str, Any]]) -> List[str]:
        """Analyze how objects are positioned relative to each other"""
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                rel = self._get_relationship(obj1, obj2)
                if rel:
                    relationships.append(f"{obj1['class']} is {rel} {obj2['class']}")
        
        return relationships
    
    def _get_relationship(self, obj1, obj2):
        """Determine spatial relationship between two objects"""
        x1, y1 = obj1['center']
        x2, y2 = obj2['center']
        
        if abs(x1 - x2) < 50:  # Vertically aligned
            if y1 < y2:
                return "above"
            else:
                return "below"
        elif abs(y1 - y2) < 50:  # Horizontally aligned
            if x1 < x2:
                return "left of"
            else:
                return "right of"
        return None
    
    def _analyze_manipulation(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze object manipulation states"""
        manipulation = {
            "held_objects": [],
            "stationary_objects": [],
            "interaction_zones": []
        }
        
        for obj in objects:
            if obj['manipulation_state'] == "possibly_held":
                manipulation["held_objects"].append(obj['class'])
            else:
                manipulation["stationary_objects"].append(obj['class'])
        
        return manipulation
    
    def _infer_scene_context(self, objects: List[Dict[str, Any]]) -> str:
        """Infer overall scene context from objects"""
        object_classes = [obj['class'] for obj in objects]
        
        if 'person' in object_classes and any(x in object_classes for x in ['book', 'phone', 'laptop']):
            return "person engaged with digital device"
        elif 'person' in object_classes and any(x in object_classes for x in ['cup', 'glass', 'bottle']):
            return "person drinking or holding beverage"
        elif 'dog' in object_classes and any(x in object_classes for x in ['ball', 'toy', 'frisbee']):
            return "dog playing with toy"
        
        return "general scene with multiple objects"
    
    def _infer_actions(self, objects: List[Dict[str, Any]], query: str = None) -> str:
        """Infer possible actions based on objects and query"""
        held_objects = [obj for obj in objects if obj['manipulation_state'] == "possibly_held"]
        
        if not held_objects:
            return "static scene with no apparent manipulation"
        
        actions = []
        for obj in held_objects:
            if obj['class'] in ['phone', 'laptop']:
                actions.append(f"using {obj['class']}")
            elif obj['class'] in ['cup', 'glass', 'bottle']:
                actions.append(f"drinking from {obj['class']}")
            elif obj['class'] in ['book', 'newspaper']:
                actions.append(f"reading {obj['class']}")
            else:
                actions.append(f"holding {obj['class']}")
        
        return ", ".join(actions)

# Global instance
vision_reasoner = None

def get_vision_reasoner():
    global vision_reasoner
    if vision_reasoner is None:
        vision_reasoner = ExtendedVisionReasoner()
    return vision_reasoner