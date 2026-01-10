import datetime
from typing import Any, Dict
from .normal_agent import NormalPersonAgent
from .blind_agent import BlindPersonAgent
from .mute_agent import MutePersonAgent
from .memory_agent import LessHearableAgent
from datetime import datetime
import torch
from torchvision import transforms
from PIL import Image
import torchaudio
from models.encoders import ImageEncoder, AudioEncoder, TextEncoder
from config.settings import DEVICE, EMBED_DIM, MEMORY_FILE, TOP_N
from .output_layer import OutputLayer
class AgentManager:
    def __init__(self, memory_manager, img_enc, aud_enc, txt_enc, device):
        self.memory_manager = memory_manager
        self.img_enc = img_enc
        self.aud_enc = aud_enc
        self.txt_enc = txt_enc
        self.device = device
        self.output_layer = OutputLayer(
    facts_path="knowledge/real_world_facts.pt",
    knowledge_path="knowledge/knowledge_embeddings.pt"
)
        # Initialize all agents
        self.agents = {
    "normal": NormalPersonAgent(
        self.memory_manager, img_enc, aud_enc, txt_enc, device,
        manager=self,
        output_layer=self.output_layer
    ),
    "blind": BlindPersonAgent(
        self.memory_manager, aud_enc, txt_enc, device,
        manager=self,                       # 👈 added
        output_layer=self.output_layer      # 👈 added
    ),
    "mute": MutePersonAgent(
        self.memory_manager, img_enc, txt_enc, device,
        manager=self,
        output_layer=self.output_layer
    )
            
        }
        
    def get_agent(self, agent_type: str):
        """Get specific agent by type"""
        return self.agents.get(agent_type, self.agents["normal"])
    
    def run_agent_analysis(self, agent_type: str, **kwargs) -> Dict[str, Any]:
        """Run analysis with specific agent"""
        agent = self.get_agent(agent_type)
        
        # Call both methods
        #results_modalities = agent.process_modalities(**kwargs)
        #results_forward = agent.process_and_forward(**kwargs)  # <-- added this call
        
        #reasoned_sentence = agent.generate_reasoned_sentence(results_modalities)
        results_forward = agent.process_and_forward(**kwargs)
        
        return {
        "agent_type": agent_type,
        "results_forward": results_forward,
        "reasoned_sentence": results_forward.get("reasoned_sentence"),
        "knowledge_result": results_forward.get("knowledge_result"),
        "timestamp": datetime.now().isoformat()
    }