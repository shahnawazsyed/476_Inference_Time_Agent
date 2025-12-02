"""
agent.py

Defines the main agent logic: Selects which reasoning strategy to apply for each input example
"""

from strategies import self_consistency, self_refine, assumption_explicit_reasoning

def run_agent(prompt: str, domain: str) -> str:
    if domain == "math":
        result = self_consistency(prompt, True)
    elif domain == "common_sense":
        result = self_consistency(prompt, False)
    elif domain == "planning" or domain == "coding":
        result = self_refine(prompt, domain)
    else: #future prediction
        result = assumption_explicit_reasoning(prompt, domain)
    return result