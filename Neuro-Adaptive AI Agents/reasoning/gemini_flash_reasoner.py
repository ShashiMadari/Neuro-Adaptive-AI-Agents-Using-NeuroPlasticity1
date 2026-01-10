import google.generativeai as genai
from typing import Dict, Any, List
import os
from dataclasses import dataclass
import json
import pyttsx3
import time


@dataclass
class GeminiFlashConfig:
    model_name: str = "models/gemini-2.5-flash"
    temperature: float = 0.1  # Low for factual reasoning
    max_output_tokens: int = 1024

class GeminiFlashReasoner:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY in .env")
        
        genai.configure(api_key=self.api_key)
        self.config = GeminiFlashConfig()
        self.model = genai.GenerativeModel(
            self.config.model_name,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
            }
        )

    # ---------------- Advanced Reasoning ----------------
    def generate_advanced_reasoning(self, query: str, cognitive_data: Dict[str, Any]) -> str:
        prompt = self._build_reasoning_prompt(query, cognitive_data)
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"⚠️ Gemini reasoning temporarily unavailable. Error: {str(e)}"

    def _build_reasoning_prompt(self, query: str, analysis: Dict[str, Any]) -> str:
        return f"""
You are a Neuro-Adaptive AI Reasoning Engine. 
Think like a human manipulating objects and concepts in their mind.
Do not describe things statically. Instead, reason dynamically about how elements interact,
what they imply, and what transformations are possible.

**QUERY**: "{query}"

**COGNITIVE ANALYSIS DATA**:
{json.dumps(analysis, indent=2, default=str)}

**REASONING TASK**:
1. Focus on causal, spatial, and temporal relationships between elements.
2. Show reasoning as if you are mentally manipulating the objects or ideas.
3. Prefer "because", "so that", "if...then" reasoning chains over flat descriptions.
4. Consider potential actions, consequences, or hidden implications.
5. Provide reasoning that flows naturally, like human thought.

**OUTPUT REQUIREMENTS**:
- Professional but accessible language
- Evidence-based conclusions
- Causal reasoning (not static summary)
- No markdown, just clean text
- 2-4 paragraphs

**Begin reasoning:**
"""

    # ---------------- Comparative Insight ----------------
    def generate_comparative_insight(self, items: List[Dict[str, Any]]) -> str:
        prompt = f"""
Compare these memory items and reason about them dynamically:

{json.dumps(items[:3], indent=2, default=str)}

Focus on:
- How the objects or ideas relate to each other
- Differences in their roles or functions
- Temporal or causal relationships
- What these comparisons imply about the bigger picture

Output natural reasoning, not just a list.
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Comparative analysis unavailable: {str(e)}"

    # ---------------- Summarization ----------------
    def summarize_reasoning(self, reasoning_traces: List[str]) -> str:
        if not reasoning_traces:
            return "No reasoning available for summarization."
        
        reasoning_text = "\n".join([f"- {trace}" for trace in reasoning_traces])
        prompt = f"""
You are a cognitive compression module for a neuro-adaptive AI system.
Summarize the following reasoning traces into a concise, meaningful conclusion.

REASONING TRACES TO SUMMARIZE:
{reasoning_text}

SUMMARY REQUIREMENTS:
- Extract the core insight or main conclusion
- 1-2 sentences maximum for compact storage
- Preserve key relationships and causal reasoning
- Use clear, professional language
- Avoid repetition and redundancy

Provide only the summary text (no bullets, no formatting, no markdown).
"""
        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            if summary.startswith('"') and summary.endswith('"'):
                summary = summary[1:-1]
            return summary
        except Exception as e:
            print(f"⚠️ Gemini summary error: {e}")
            fallback = reasoning_traces[0] if reasoning_traces else "No reasoning available"
            return fallback[:200] + "..." if len(fallback) > 200 else fallback

    # ---------------- Human-Like Explanation ----------------
    def generate_human_like_explanation(
        self, query: str, facts: List[Dict[str, Any]], knowledge: List[Dict[str, Any]], agent_type: str = "normal"
    ) -> Dict[str, Any]:

        # Format facts safely
        facts_str = "None"
        if facts and isinstance(facts, list):
            facts_lines = []
            for f in facts:
                if isinstance(f, dict):
                    for k, v in f.items():
                        facts_lines.append(f"- {k}: {v}")
                else:
                    facts_lines.append(f"- {str(f)}")
            facts_str = "\n".join(facts_lines)

        # Format knowledge safely
        knowledge_str = "None"
        if knowledge and isinstance(knowledge, list):
            knowledge_lines = []
            for item in knowledge:
                if isinstance(item, dict):
                    for name, details in item.items():
                        section = f"- {details.get('name', name)}"
                        if isinstance(details, dict):
                            if "statement" in details:
                                section += f"\n  • Statement: {details['statement']}"
                            if "formula" in details:
                                section += f"\n  • Formula: {details['formula']}"
                            if "explanation" in details:
                                section += f"\n  • Explanation: {details['explanation']}"
                            if "example" in details:
                                section += f"\n  • Example: {details['example']}"
                        knowledge_lines.append(section)
                else:
                    knowledge_lines.append(f"- {str(item)}")
            knowledge_str = "\n".join(knowledge_lines)

        # ✅ Print facts and knowledge
        print(f"Facts for explanation:\n{facts_str}\n")
        print(f"Knowledge for explanation:\n{knowledge_str}\n")
        print(f"Agent query: {query}\n")

        # Prepare prompt
        prompt = f"""
You are a human-like teacher explaining concepts naturally.

QUERY: {query}

FACTS (must be included in notes):
{facts_str}

KNOWLEDGE (useful background to expand the explanation):
{knowledge_str}

OUTPUT FORMAT (STRICT):
TEXT: <short, compact notes for quick reading>
AUDIO: <spoken explanation in natural step-by-step teaching style>

EXAMPLE OUTPUT:
TEXT: Energy equals mass times the square of the speed of light.
AUDIO: Let me explain. Einstein showed that mass and energy are deeply connected...
       If you take a tiny amount of matter and multiply it by the speed of light squared,
       you get an enormous amount of energy. This explains why nuclear reactions release so much power.

Now generate the explanation for the current query. Do not skip either section.
"""

        try:
            response = self.model.generate_content(prompt)
            reasoning = response.text.strip()

            # Robust parsing
            text_part, audio_part = None, None
            if "TEXT:" in reasoning and "AUDIO:" in reasoning:
                parts = reasoning.split("AUDIO:")
                text_part = parts[0].replace("TEXT:", "").strip()
                audio_part = parts[1].strip()
            else:
                for line in reasoning.splitlines():
                    if line.startswith("TEXT:"):
                        text_part = line.replace("TEXT:", "").strip()
                    elif line.startswith("AUDIO:"):
                        audio_part = line.replace("AUDIO:", "").strip()

            result = {}
            if agent_type == "normal":
                if text_part:
                    result["text"] = text_part
                if audio_part:
                    result["audio"] = self._text_to_audio(audio_part, agent_type="normal")
            #elif agent_type == "blind":
                #if audio_part:
                  #  result["audio"] = self._text_to_audio(audio_part)
            #elif agent_type == "mute":
               # if text_part:
                   # result["text"] = text_part'''
            else:
                result["text"] = reasoning

            return result

        except Exception as e:
            return {"error": f"⚠️ Human-like explanation unavailable: {str(e)}"}

    # ---------------- Text-to-Audio ----------------
    def _text_to_audio(self, text: str, agent_type: str = "normal") -> str:
        """Convert text to audio, save it in static/audio/, and return the relative URL path."""
        try:
            import os, time, pyttsx3

            # Create static/audio folder if it doesn't exist
            audio_dir = os.path.join("static", "audio")
            os.makedirs(audio_dir, exist_ok=True)

            # Create unique filename
            timestamp = int(time.time() * 1000)
            filename = f"{agent_type}_audio_{timestamp}.mp3"
            filepath = os.path.join(audio_dir, filename)

            # Generate and save the audio
            engine = pyttsx3.init()
            engine.save_to_file(text, filepath)
            engine.runAndWait()

            # Return relative URL path for browser playback
            return f"/static/audio/{filename}"

        except Exception as e:
            print(f"⚠️ Audio synthesis failed: {e}")
            return ""

    # ---------------- Blind Agent Human-Like Audio ----------------
    def generate_blind_audio_explanation(
        self, query: str, facts: List[Dict[str, Any]], knowledge: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate ONLY audio explanation for BlindPersonAgent.
        More detailed, quick-flowing spoken style.
        """
        # Format facts
        facts_str = "None"
        if facts and isinstance(facts, list):
            facts_lines = []
            for f in facts:
                if isinstance(f, dict):
                    for k, v in f.items():
                        facts_lines.append(f"- {k}: {v}")
                else:
                    facts_lines.append(f"- {str(f)}")
            facts_str = "\n".join(facts_lines)

        # Format knowledge
        knowledge_str = "None"
        if knowledge and isinstance(knowledge, list):
            knowledge_lines = []
            for item in knowledge:
                if isinstance(item, dict):
                    for name, details in item.items():
                        section = f"- {details.get('name', name)}"
                        if isinstance(details, dict):
                            if "statement" in details:
                                section += f"\n  • {details['statement']}"
                            if "formula" in details:
                                section += f"\n  • Formula: {details['formula']}"
                            if "explanation" in details:
                                section += f"\n  • {details['explanation']}"
                            if "example" in details:
                                section += f"\n  • Example: {details['example']}"
                        knowledge_lines.append(section)
                else:
                    knowledge_lines.append(f"- {str(item)}")
            knowledge_str = "\n".join(knowledge_lines)

        # Prompt designed for blind-agent audio
        prompt = f"""
You are explaining to a blind person using only voice.
Do not give written notes. 
Your style must be quick, natural, and conversational.
Break down the explanation step by step, almost like storytelling.
Avoid pauses that feel like "reading". Flow like you're talking directly.

QUERY: {query}

FACTS:
{facts_str}

KNOWLEDGE:
{knowledge_str}

OUTPUT FORMAT (STRICT):
AUDIO: <detailed spoken explanation only more detailed than usual>

Example style:
"Aha, let me guide you. Imagine holding a small piece of matter...
if you could squeeze it into pure energy, it would explode with massive power,
because Einstein’s formula shows mass hides huge energy inside."

Now generate the AUDIO output only.
"""

        try:
            response = self.model.generate_content(prompt)
            reasoning = response.text.strip()

            audio_part = None
            if "AUDIO:" in reasoning:
                audio_part = reasoning.split("AUDIO:")[1].strip()
            else:
                audio_part = reasoning

            # Convert to speech
            audio_file = self._text_to_audio(audio_part, agent_type="blind")

            return {
                "audio_text": audio_part,  # keep raw transcript too
                "audio_file": audio_file
            }

        except Exception as e:
            return {"error": f"⚠️ Blind audio explanation unavailable: {str(e)}"}
    # ---------------- Mute Agent Human-Like Text ----------------
    def generate_mute_text_explanation(
        self, query: str, facts: List[Dict[str, Any]], knowledge: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate ONLY text explanation for MutePersonAgent.
        Clear, concise, note-like text without audio.
        """
        # Format facts
        facts_str = "None"
        if facts and isinstance(facts, list):
            facts_lines = []
            for f in facts:
                if isinstance(f, dict):
                    for k, v in f.items():
                        facts_lines.append(f"- {k}: {v}")
                else:
                    facts_lines.append(f"- {str(f)}")
            facts_str = "\n".join(facts_lines)

        # Format knowledge
        knowledge_str = "None"
        if knowledge and isinstance(knowledge, list):
            knowledge_lines = []
            for item in knowledge:
                if isinstance(item, dict):
                    for name, details in item.items():
                        section = f"- {details.get('name', name)}"
                        if isinstance(details, dict):
                            if "statement" in details:
                                section += f"\n  • {details['statement']}"
                            if "formula" in details:
                                section += f"\n  • Formula: {details['formula']}"
                            if "explanation" in details:
                                section += f"\n  • {details['explanation']}"
                            if "example" in details:
                                section += f"\n  • Example: {details['example']}"
                        knowledge_lines.append(section)
                else:
                    knowledge_lines.append(f"- {str(item)}")
            knowledge_str = "\n".join(knowledge_lines)

        # Prompt designed for mute-agent text
        prompt = f"""
You are explaining like a mute person who uses only written text.
Your style must be:
- Short, structured, and clear
- Step-by-step explanation
- Use bullet points, simple notes, or compact sentences
- Avoid storytelling tone, focus on clarity
- No audio references, only text

QUERY: {query}

FACTS:
{facts_str}

KNOWLEDGE:
{knowledge_str}

OUTPUT FORMAT (STRICT):
TEXT: <compact written explanation only>
"""

        try:
            response = self.model.generate_content(prompt)
            reasoning = response.text.strip()

            text_part = None
            if "TEXT:" in reasoning:
                text_part = reasoning.split("TEXT:")[1].strip()
            else:
                text_part = reasoning

            return {
                "text": text_part
            }

        except Exception as e:
            return {"error": f"⚠️ Mute text explanation unavailable: {str(e)}"}

    # ---------------- Neuro-Adaptive Feedback Generator ----------------
    def generate_agent_feedback(
        self,
        agent_from: str,
        agent_to: str,
        input_text: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generates neuro-adaptive feedback between agents.
        Example:
            - From mute → blind  (text → audio explanation)
            - From blind → mute  (audio text → refined text notes)
        """

        if not input_text:
            return {"error": "⚠️ No input text available for feedback generation."}

        context_str = json.dumps(context or {}, indent=2)

        # Step 1: Build dynamic prompt
        prompt = f"""
You are part of a Neuro-Adaptive Multi-Agent System.

Two agents are exchanging sensory feedback:
- Source Agent: {agent_from}
- Target Agent: {agent_to}

SOURCE OUTPUT:
\"\"\"{input_text}\"\"\"

CONTEXT (for cognitive grounding):
{context_str}

TASK:
1. If the source is 'mute' and the target is 'blind':
   - Convert the source's written reasoning into a natural, audio-friendly explanation.
   - Focus on tone, pacing, and descriptive storytelling (since the blind agent learns via audio).
   - Output format must include only AUDIO: <spoken explanation>

2. If the source is 'blind' and the target is 'mute':
   - Convert the audio explanation transcript into compact, bullet-like text reasoning.
   - Keep it factual, clear, and easy to read (no storytelling).
   - Output format must include only TEXT: <written feedback>

3. Always maintain semantic alignment: the feedback should preserve the same factual meaning,
   but adapt its expression to the target agent’s modality.

OUTPUT REQUIREMENTS:
- STRICT format (TEXT: ... or AUDIO: ...)
- Preserve reasoning quality
- Keep explanation concise and human-like
"""

        try:
            response = self.model.generate_content(prompt)
            reasoning = response.text.strip()
            #print(f"Raw feedback reasoning:\n{reasoning}\n")
            # Step 2: Parse output properly
            result = {}
            if agent_from == "mute" and agent_to == "blind":
                # Expect AUDIO output
                audio_text = reasoning.split("AUDIO:")[-1].strip() if "AUDIO:" in reasoning else reasoning
                audio_file = self._text_to_audio(audio_text, agent_type="blind")
                result = {
                    "from": agent_from,
                    "to": agent_to,
                    "audio_text": audio_text,
                    "audio_file": audio_file
                }

            elif agent_from == "blind" and agent_to == "mute":
                # Expect TEXT output
                text_part = reasoning.split("TEXT:")[-1].strip() if "TEXT:" in reasoning else reasoning
                result = {
                    "from": agent_from,
                    "to": agent_to,
                    "text": text_part
                }

            else:
                result = {"error": f"⚠️ Unsupported agent feedback direction: {agent_from} → {agent_to}"}

            return result

        except Exception as e:
            return {"error": f"⚠️ Feedback generation failed: {str(e)}"}



# ---------------- Singleton ----------------
_flash_reasoner = None

def get_flash_reasoner(api_key: str = None) -> GeminiFlashReasoner:
    global _flash_reasoner
    if _flash_reasoner is None:
        _flash_reasoner = GeminiFlashReasoner(api_key)
    return _flash_reasoner
