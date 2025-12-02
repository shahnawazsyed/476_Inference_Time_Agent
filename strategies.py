"""
strategies.py

Implements multiple inference-time reasoning algorithms.

Each function here represents one of the following reasoning techniques:
Chain of Thought prompting
Self Consistency
Self Refinement
Assumpion Explicit Reasoning

DOMAINS: math, common sense, future prediction, coding, planning

"""
from api import call_model_chat_completions
import random
from concurrent.futures import ThreadPoolExecutor, as_completed



def convertToPlainText(prompt: str): #clean this up
    conversion_sys_prompt = """
            You are a LaTeX→PlainText converter whose job is to take a user prompt that may contain mathematical expressions written in LaTeX and produce a single, unambiguous, machine-friendly plain-English representation of the math. The output will be fed back to a solver, so do not evaluate, simplify, or solve anything — only convert notation to clear words and ASCII-like tokens. Follow these rules exactly:
            1. Output format
            - Return only the converted plain text (no explanations, no extra commentary, no Markdown, no LaTeX).  
            - Keep punctuation (commas, parentheses) needed for structure.
            2. General guidelines
            - Preserve variable names exactly (case-sensitive): x → x, X → X.
            - Preserve ordering and grouping with words and parentheses where needed.
            - Use words for operators and relation symbols (see mapping below).
            - For subscripts and superscripts, use readable phrases (see examples).
            - Spell Greek letters and standard symbols (e.g., \\alpha → alpha, \\pi → pi).
            - Do not attempt to compute numeric values or simplify algebraic expressions.
            - Do not add commentary like “this means” or “note that”.
            3. Preferred verbal mappings
            - + → plus; - → minus; \\times or \\cdot → times; \\div or / → divided by or over (use over for fractions).
            - = → equals; \\neq → not equal to; \\le / \\leq → less than or equal to; \\ge / \\geq → greater than or equal to.
            - Superscript: x^2 → x squared OR x to the power of 2; x^{n+1} → x to the power of (n plus 1).
            - Subscript: a_i → a sub i; x_{ij} → x sub i j.
            - Fractions: \\frac{a}{b} → a over b (or a divided by b); keep numerator/denominator grouping with parentheses when complex: ((a plus b) over (c minus d)).
            - Summation/product: \\sum_{i=1}^n a_i → sum from i = 1 to n of a sub i; \\prod → product from ... of ....
            - Limits: \\lim_{x\\to 0} f(x) → limit as x approaches 0 of f(x).
            - Integrals: \\int_a^b f(x)\\,dx → integral from a to b of f(x) dx.
            - Derivatives: \\frac{d}{dx} f(x) → derivative of f(x) with respect to x; f'(x) → f prime of x.
            - Partial derivatives: \\frac{\\partial f}{\\partial x} → partial derivative of f with respect to x.
            - Functions: keep common names: \\sin x → sin(x), \\ln x → ln(x).
            - Sets and logic: \\in → in; \\notin → not in; \\forall x → for all x; \\exists → there exists.
            - Matrices: \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix} → 2 by 2 matrix with rows [a, b] and [c, d].
            - Parentheses/brackets: keep them as parentheses in text for clarity: use ( ) and say words inside as needed.
            4. Ambiguity handling
            - If an expression has grouping, reflect grouping explicitly using parentheses and the word of where helpful: e.g., \\frac{1}{1+x^2} → 1 over (1 plus x squared).
            - If a symbol is ambiguous in context (e.g., letter e might be Euler’s number), do not guess — output e (preserve as-is). The solver will interpret.
            5. Preserve non-math text
            - Keep surrounding normal-language text as-is, but convert any LaTeX math segments inline or display math into the plain-text math representation.
            - Maintain sentence structure and punctuation so the converted prompt can be fed back unchanged except for math notation.
            6. Do not translate
            - Do not translate code blocks, filenames, or computer-language snippets unless they are mathematical expressions in LaTeX.
            7. Examples
            - Input (user): Solve \\frac{d}{dx}\\left(x^2 \\sin x\\right)=0 for x
                - Output: Solve derivative of (x squared times sin(x)) with respect to x equals 0 for x
            - Input: Compute \\int_0^{\\pi/2} \\sin^2 x\\,dx
                - Output: Compute integral from 0 to pi over 2 of sin squared x dx
            - Input: Find eigenvalues of \\begin{pmatrix}2 & 1\\\\1 & 2\\end{pmatrix}
                - Output: Find eigenvalues of 2 by 2 matrix with rows [2, 1] and [1, 2]
            - Input: If \\sum_{n=1}^\\infty a_n converges, show \\lim_{n\\to\\infty} a_n = 0.
                - Output: If sum from n = 1 to infinity of a sub n converges, show limit as n approaches infinity of a sub n equals 0.
            - Input: Solve x^2 + y^2 = 1
                - Output: Solve x squared plus y squared equals 1
            - Input: Let f(x)=\\ln(x^2+1). Compute f'(x).
                - Output: Let f(x) = ln(x squared plus 1). Compute f prime of x.
            Act exactly as above for every user message. Always convert math to plain text without solving or commenting.
            """

    ans = call_model_chat_completions(prompt=prompt, system=conversion_sys_prompt, max_tokens=4096)["text"]
    return ans.strip() if ans is not None else ""

def self_consistency(prompt: str, isMath: bool = False, num_samples: int = 7, verbose=False): #runs CoT in parallel
    results = {}
    if isMath:
        prompt = convertToPlainText(prompt) 
    with ThreadPoolExecutor(max_workers=num_samples) as executor: #simulataneous API calls
        future_to_ans = {
            executor.submit(chain_of_thought, prompt, random.uniform(0.5, 1.0)): i #randomized temp
            for i in range(num_samples)
        }
        for future in as_completed(future_to_ans):
            ans = future.result()   
            if ans in results:
                results[ans] += 1
            else:
                results[ans] = 1
    if verbose:
        print("\nnum of unique results", len(results))
    if results:
        majority_ans = max(results, key=results.get)
        return majority_ans
    return ""

def chain_of_thought(prompt: str, temp: float = 0.0) -> str:
    cot_instruction = (
        "Think through this problem step by step and solve it completely. "
        "You must provide a complete solution, not just validate or critique. "
        "At the very end, write 'Final Answer:' followed by your complete answer."
    )
    cot_system_prompt = "You are a problem-solving assistant. Always provide complete solutions."
    reasoning_resp = call_model_chat_completions(prompt=prompt, system=cot_system_prompt+" "+cot_instruction, max_tokens=4096, temperature=temp)["text"]
    extract_answer_system_prompt = (
        "Extract the complete final answer from this solution. "
        "For plans, extract all the steps. For numerical answers, extract just the number. "
        "Reply with only the answer itself."
    )
    answer = call_model_chat_completions(prompt=reasoning_resp, system=extract_answer_system_prompt, max_tokens=256, temperature=temp)["text"]
    return answer.strip() if answer is not None else ""



def self_refine(prompt: str, domain: str, temp: float = 0.0, max_iter=3, verbose=False) -> str:
    initial_ans = call_model_chat_completions(prompt=prompt, max_tokens=4096, temperature=temp)["text"]
    refine_sys_prompt = f"You are a critical evaluator specializing in {domain}. Review the answer provided to the following prompt: {prompt} and give constructive feedback on how to improve it. Focus on accuracy, completeness, clarity, and relevance to {domain}. Point out any errors, missing information, or areas that need better explanation. Be specific about what needs improvement. Do not provide a revised answer, only feedback."
    new_ans = initial_ans
    for _ in range (max_iter): #3 calls per iteration
        feedback = call_model_chat_completions(prompt=new_ans, system=refine_sys_prompt, max_tokens=2048, temperature=temp)["text"]
        sentiment_prompt = f"Rate the sentiment of this feedback with respect to how correct and high-quality the answer is, from -1 (very negative, many issues) to 1 (very positive, excellent answer). Return only a number (float):\n\n{feedback}"
        sentiment_score = float(call_model_chat_completions(prompt=sentiment_prompt, max_tokens=16, temperature=0.0)["text"].strip())
        if verbose:
            print("\nsentiment: ", sentiment_score)
        if sentiment_score >= 0.7:
            break 
        SYS_PROMPT = "You are a helpful assistant. Provide the requested revised answer clearly and concisely. Do not include any of the previous prompts, answers, or feedback in your response."
        formatted_feedback = (
            f"Please revise the following answer based on the feedback provided:\n\n"
            f"ORIGINAL PROMPT: {prompt}\n"
            f"PREVIOUS ANSWER:\n{new_ans}\n\n"
            f"FEEDBACK:\n{feedback}\n\n"
            f"Provide a REVISED, complete answer now that addresses all points of the feedback."
        )
        new_ans = call_model_chat_completions(prompt=formatted_feedback, system=SYS_PROMPT, max_tokens=4096, temperature=temp)["text"]
    return new_ans

def assumption_explicit_reasoning(prompt: str, domain: str, temp: float = 0.0) -> str:
    init_ans = chain_of_thought(prompt, temp) #2
    extraction_sys_prompt = (
        f"You are a critical evaluator specializing in {domain}. our task is to read the provided draft answer and extract all implicit assumptions that must be true for that answer to hold. Return only assumptions. "
    "Each assumption must be explicit, minimal, falsifiable, and phrased as a concrete condition about the world, data, behavior, or model inputs. Do not justify or evaluate them."
    "Do not restate the draft answer."
    "List assumptions as numbered items."
    )
    assumptions = call_model_chat_completions(prompt=init_ans, system=extraction_sys_prompt, max_tokens=1024)["text"] #3
    reasoning_sys_prompt = (
        f"You are a {domain} expert. Use the following assumptions to construct your solution. "
        "Think step-by-step. At the end of your solution, write 'Final Answer:' followed by the final, concise result. "
        "Do not include any other commentary, and DO NOT list the assumptions separately. "
        f"Assumptions: {assumptions}"
    )
    reasoning_resp = call_model_chat_completions(
        prompt=prompt,                 
        system=reasoning_sys_prompt, 
        max_tokens=4096, 
        temperature=temp
    )["text"]
    extract_answer_system_prompt = (
        "Extract the complete final answer from this solution. "
        "Reply with only the final answer itself—no explanations, no commentary, and no preceding text."
    )
    final_answer = call_model_chat_completions(
        prompt=reasoning_resp, 
        system=extract_answer_system_prompt, 
        max_tokens=256, 
        temperature=temp
    )["text"]
    
    return final_answer.strip()

