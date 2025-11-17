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
    cot_instruction = (
        "When answering, reason step by step before stating the final answer. "
        "At the end, clearly mark your final answer with 'Final Answer:'."
    )
    cot_system_prompt = "You are an advanced reasoning assistant."
    reasoning_resp = call_model_chat_completions(prompt=prompt, system=cot_system_prompt+" "+cot_instruction)["text"]
    extract_answer_prompt = (
        f"Here is a reasoning process:\n\n{reasoning_resp}\n\n"
        "Extract and return ONLY the final answer, with no additional text or explanation."
    )
    answer = call_model_chat_completions(prompt=extract_answer_prompt)["text"]
    return answer