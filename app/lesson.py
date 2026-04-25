import json
import os
from langchain_core.prompts import PromptTemplate
from app.llm import llm

def generate_lesson_from_architecture():
    # Read architecture content
    arch_path = os.path.join(os.path.dirname(__file__), "..", "ARCHITECTURE.md")
    try:
        with open(arch_path, "r") as f:
            arch_content = f.read()
    except Exception as e:
        arch_content = "Architecture documentation not found."

    prompt = PromptTemplate.from_template("""
You are an expert teacher who creates engaging, interactive lessons.

You MUST:
- Follow Gagné’s Nine Events
- Apply Merrill’s First Principles
- Teach for deep understanding, not memorization

STRICT RULES:
- Output ONLY valid JSON
- Break content into small, interactive blocks
- Avoid long paragraphs
- Use curiosity, questions, and tasks
- Make it beginner-friendly but conceptually deep

CRITICAL:
- Explain WHY, not just WHAT
- Include prediction questions
- Include common mistakes
- Force active thinking every few steps

OUTPUT FORMAT:
{{
  "lesson": [
    {{
      "type": "attention",
      "content": "Engaging hook using real-world problem"
    }},
    {{
      "type": "objective",
      "content": "Clear outcome of lesson"
    }},
    {{
      "type": "recall",
      "question": "Connect to prior knowledge"
    }},
    {{
      "type": "prediction",
      "question": "Ask what student thinks will happen"
    }},
    {{
      "type": "concept",
      "title": "...",
      "explanation": "Explain WHY concept matters"
    }},
    {{
      "type": "example",
      "content": "Simple real-world or code example"
    }},
    {{
      "type": "guided_practice",
      "steps": ["Step 1", "Step 2", "Step 3"]
    }},
    {{
      "type": "active_task",
      "task": "Hands-on problem"
    }},
    {{
      "type": "feedback",
      "common_mistake": "...",
      "fix": "Clear correction"
    }},
    {{
      "type": "assessment",
      "question": "Conceptual + reasoning-based question"
    }},
    {{
      "type": "extension",
      "task": "Real-world or advanced variation"
    }}
  ]
}}

INPUT BLUEPRINT:
{blueprint}
""")
    
    chain = prompt | llm
    response = chain.invoke({"blueprint": arch_content})
    
    import re
    content = response.content
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)
        
    try:
        return json.loads(content)
    except Exception as e:
        print("Failed to parse JSON:", content)
        raise e
