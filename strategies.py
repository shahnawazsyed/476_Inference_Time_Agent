"""
strategies.py

Implements multiple inference-time reasoning algorithms.

Each function here represents one of the following reasoning techniques:
chain of thought prompting

self consistency -- maybe for math, common sense,

"""

from api import call_model_chat_completions

def chain_of_thought(prompt: str) -> str: #could be good for planning, coding, future prediction (?)
    """
    Chain-of-Thought (CoT) inference strategy.
    Encourages the model to reason step by step before extracting a deterministic answer.
    """
    #print(prompt)
    cot_instruction = (
        "Think through this problem step by step and solve it completely. "
        "You must provide a complete solution, not just validate or critique. "
        "At the very end, write 'Final Answer:' followed by your complete answer."
    )
    cot_system_prompt = "You are a problem-solving assistant. Always provide complete solutions."
    reasoning_resp = call_model_chat_completions(prompt=prompt, system=cot_system_prompt+" "+cot_instruction, max_tokens=8196)["text"]
    #print(reasoning_resp)
    extract_answer_system_prompt = (
        "Extract the complete final answer from this solution. "
        "For plans, extract all the steps. For numerical answers, extract just the number. "
        "Reply with only the answer itself."
    )
    answer = call_model_chat_completions(prompt=reasoning_resp, system=extract_answer_system_prompt, max_tokens=2048)["text"]
    #print(answer)
    return answer