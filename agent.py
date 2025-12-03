"""
agent.py

Defines the main agent logic: Selects which reasoning strategy to apply for each input example
"""

from strategies import self_consistency, self_refine, assumption_explicit_reasoning, chain_of_thought, get_domain, convertToPlainText
def run_agent(prompt: str) -> str:
    domain = get_domain(prompt) #+1 call
    #print("Domain", domain)
    # if domain not in ("Planning", "Coding", "Math", "Common Sense", "Future Prediction"):
    #     print("Domain bad")
    if domain == "Math":
        result = self_consistency(prompt, True) #15 calls max
    elif domain == "Common Sense":
        result = self_consistency(prompt, False) #14 calls max
    elif domain == "Planning" or domain == "Coding":
        #result = chain_of_thought(prompt)
        result = self_refine(prompt, domain) #10 calls max
    elif domain == "Future Prediction": #future prediction
        result = assumption_explicit_reasoning(prompt, domain) #5 calls max
    else: #fallback -> self consistency
        result = self_consistency(prompt) #14 calls max
    if result == "": #fallback for empty output, reduce chance of "" again with multiple CoT samples thru self consistency
        #no point in re running Self consistency with math or common sense since we originally did
        if domain == "Planning" or domain == "Coding":
            new_res = self_consistency(prompt, num_samples=3) #3 * 2 = 6 more API calls --> 17 total, ensures majority vote with odd #
        elif domain == "Future Prediction":
            new_res = self_consistency(prompt, num_samples=5) #5 * 2 = 10 more API calls --> 16 total
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