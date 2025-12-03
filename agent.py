"""
agent.py

Defines the main agent logic: Selects which reasoning strategy to apply for each input example
"""

from strategies import self_consistency, self_refine, assumption_explicit_reasoning, chain_of_thought, get_domain
def run_agent(prompt: str) -> str:
    domain = get_domain(prompt)
    #print("Domain", domain)
    # if domain not in ("Planning", "Coding", "Math", "Common Sense", "Future Prediction"):
    #     print("Domain bad")
    if domain == "Math":
        result = self_consistency(prompt, True) #8 calls
    elif domain == "Common Sense":
        result = self_consistency(prompt, False) #7 calls
    elif domain == "Planning" or domain == "Coding":
        #result = chain_of_thought(prompt)
        result = self_refine(prompt, domain) #10 calls
    elif domain == "Future Prediction": #future prediction
        result = assumption_explicit_reasoning(prompt, domain) #5 calls
    else: #fallback -> CoT
        result = chain_of_thought(prompt)
    if result == "": #fallback again
        new_res = self_consistency(prompt) #reduce chance of "" with multiple CoT samples
        # debugging for empty answers:
        # if new_res == "":
        #     #print("Still empty, prompt:", prompt)
        #     print("Domain:", domain)
        # else:
        #     print("fixed")
        #if new_res == "":
            #print("Still empty")
        return new_res
    return result