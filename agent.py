"""
agent.py

Defines the main agent logic:
- Selects which reasoning strategy to apply for each input example.
- Manages prompts, responses, and postprocessing.
- Calls `call_model_chat_completions()` from api.py indirectly via strategies.
"""

from strategies import chain_of_thought

def run_agent(prompt: str) -> str:
    #TODO: add decisioning for which reasoning strategy to use
    result = chain_of_thought(prompt)
    return result