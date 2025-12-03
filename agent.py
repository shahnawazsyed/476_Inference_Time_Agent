"""
agent.py

Defines the main agent logic: Selects which reasoning strategy to apply for each input example
"""

from strategies import self_consistency, self_refine, assumption_explicit_reasoning

def run_agent(prompt: str, domain: str) -> str:
    if domain == "math":
        result = self_consistency(prompt, True) #8 calls
    elif domain == "common_sense":
        result = self_consistency(prompt, False) #7 calls
    elif domain == "planning" or domain == "coding":
        result = self_refine(prompt, domain) #10 calls
    else: #future prediction
        result = assumption_explicit_reasoning(prompt, domain) #5 calls
    return result