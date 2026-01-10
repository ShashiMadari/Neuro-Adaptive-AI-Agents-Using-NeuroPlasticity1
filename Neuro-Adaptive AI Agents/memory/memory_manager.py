import json
import os
from typing import List, Dict, Any

class MemoryManager:
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
    
    def load_memory(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.memory_file):
            return []
        with open(self.memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def save_memory(self, memory: List[Dict[str, Any]]):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    
    def add_to_memory(self, new_entry: Dict[str, Any]):
        memory = self.load_memory()
        memory.append(new_entry)
        self.save_memory(memory)