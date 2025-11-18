"""
agent.py

Defines the main agent logic:
- Selects which reasoning strategy to apply for each input example.
- Manages prompts, responses, and postprocessing.
- Calls `call_model_chat_completions()` from api.py indirectly via strategies.
"""

from strategies import chain_of_thought

def run_agent(prompt: str, domain: str) -> str:
    #TODO: add decisioning for which reasoning strategy to use
    if domain == "math":
        isMath = True
    else:
        isMath = False
    #print(isMath)
    result = chain_of_thought(prompt, isMath)
    return result