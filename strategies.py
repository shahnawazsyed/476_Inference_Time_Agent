"""
strategies.py

Implements multiple inference-time reasoning algorithms.

Each function here represents one of the following reasoning techniques:
Chain of Thought prompting
Self Consistency
Self Refinement
Assumption Explicit Reasoning

"""
import re
from api import call_model_chat_completions
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_domain(prompt: str):
    sys_prompt = (
        "You are a helpful assistant. Analyze the given prompt and determine its topic domain from the following options: Math, Common Sense, Future Prediction, Coding, Planning\n"
        "If the domain is not listed in the given options, choose the BEST option from these: Math, Common Sense, Future Prediction, Coding, Planning. DO NOT make up your own domain or decide one that is not listed in the given options..\n"
        "DO NOT attempt to answer the prompt. Your answer should be one of the following (case sensitive) based on the topic of the prompt: Math, Common Sense, Future Prediction, Coding, Planning.")
    full_response = call_model_chat_completions(prompt=prompt, system=sys_prompt, max_tokens=32)
    res = full_response.get("text", "")
    #print("Extracted text:", repr(res))
    # if not res:
    #     print("Full response:", full_response)
    return res.strip() if res else ""

def convertToPlainText(prompt: str): #convert from Latex to plain text
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
            6. Examples
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
        prompt = convertToPlainText(prompt) # 1 call
    with ThreadPoolExecutor(max_workers=2) as executor: #simulataneous API calls
        future_to_ans = { #each CoT = 2 max
            executor.submit(chain_of_thought, prompt, random.uniform(0.5, 1.0), isMath=isMath): i #randomized temp
            for i in range(num_samples)
        }
        for future in as_completed(future_to_ans):
            ans = future.result()
            if ans and ans != "": #avoid ""
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

def extract_final_answer(ans: str, isMath: bool = False) -> str:
    if not ans:
        return ""
    ans = ans.strip()
    match = re.search(r'(?:final\s+answer\s*:?\s*)(.*)', ans, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        answer = answer.strip('"\'')
        if answer: 
            return answer
    mc_patterns = [ #multiple choice
        r'(?:answer is|correct answer is|therefore|thus|option|answer choice)\s*:?\s*([A-E])\b',
        r'\b([A-E])\s+is\s+(?:the\s+)?correct',
        r'(?:choose|select)\s+(?:option\s+)?([A-E])\b'
    ]
    for pattern in mc_patterns:
        match = re.search(pattern, ans, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r'\b([A-E])\s*[.)]?\s*$', ans, re.IGNORECASE) #answer ends with a letter
    if match:
        return match.group(1).upper()
    if re.search(r'\d', ans): #worst case extract last number
        numbers = re.findall(r'-?\d+\.?\d*', ans)
        if numbers:
            if isMath: #assuming its a math problem
                return numbers[-1]
    return ans

def chain_of_thought(prompt: str, temp: float = 0.0, isMath: bool = False) -> str:
    cot_instruction = (
        "Think through this problem step by step and solve it completely. "
        "You must provide a complete solution, not just validate or critique. "
    )
    if isMath:
        cot_instruction += (
            "For mathematical expressions, use plain text notation instead of LaTeX. "
            "For example, use 'x squared' instead of 'x^2', 'a over b' instead of '\\frac{a}{b}'. "
        )
    cot_instruction += "At the very end, write 'Final Answer:' followed by your complete answer."
    cot_system_prompt = "You are a problem-solving assistant. Always provide complete solutions."
    reasoning_resp = call_model_chat_completions(prompt=prompt, system=cot_system_prompt+" "+cot_instruction, max_tokens=4096, temperature=temp)["text"]
    # if reasoning_resp == "":
    #     print("EMPTY REASONING")
    final_ans = extract_final_answer(reasoning_resp, isMath=isMath)
    if len(final_ans) > 500 or final_ans == reasoning_resp: #i.e. extraction didnt work
        extract_answer_system_prompt = (
            "Extract the complete final answer from this solution. "
            "For multiple choice questions, return ONLY the letter (A, B, C, D, or E). "
            "For numerical answers, return just the number. "
            "For plans or lists, extract all the steps. "
            "For reasoning questions, extract the conclusion. "
            "Reply with only the final answer itself—no explanations, no commentary.\n"
            "If you cannot determine a clear final answer, return the last meaningful statement.\n"
            "--- Examples ---\n"
            "INPUT 1: The total cost is $25, and the tax is $2.50. Final Answer: 27.50\n"
            "OUTPUT 1: 27.50\n"
            "INPUT 2: The required steps are: 1. Collect data. 2. Analyze. 3. Finalize. Final Answer: 1. Collect data. 2. Analyze. 3. Finalize\n"
            "OUTPUT 2: 1. Collect data. 2. Analyze. 3. Finalize"
        )
        answer = call_model_chat_completions(prompt=reasoning_resp, system=extract_answer_system_prompt, max_tokens=512, temperature=0.0)["text"]
        if answer and answer.strip():
            return answer.strip()
    return final_ans if final_ans else reasoning_resp.strip() if reasoning_resp is not None else "" #absolute worst case fallback

def self_refine(prompt: str, domain: str, temp: float = 0.0, max_iter=3, verbose=False) -> str:
    initial_ans = call_model_chat_completions(prompt=prompt, max_tokens=4096, temperature=temp)["text"] #1
    refine_sys_prompt = f"You are a critical evaluator specializing in {domain}. Review the answer provided to the following prompt: {prompt} and give constructive feedback on how to improve it. Focus on accuracy, completeness, clarity, and relevance to {domain}. Point out any errors, missing information, or areas that need better explanation. Be specific about what needs improvement. Do not provide a revised answer, only feedback."
    new_ans = initial_ans
    for _ in range (max_iter): #3 calls per iteration = 9 total by default
        feedback = call_model_chat_completions(prompt=new_ans, system=refine_sys_prompt, max_tokens=2048, temperature=temp)["text"]
        sentiment_prompt = f"Rate the sentiment of this feedback with respect to how correct and high-quality the answer is, from -1 (very negative, many issues) to 1 (very positive, excellent answer). Return ONLY a single number between -1 and 1 as a decimal (e.g., 0.7, -0.3, 0.9). Do not include any other text or explanation.\n\nFeedback:\n{feedback}"
        sentiment_result = call_model_chat_completions(prompt=sentiment_prompt, max_tokens=16, temperature=0.0)["text"]
        sentiment_text = sentiment_result.strip() if sentiment_result is not None else ""
        match = re.search(r'-?\d+\.?\d*', sentiment_text)
        if match:
            sentiment_score = float(match.group())
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        else:
            sentiment_score = 0.0
        if verbose:
            print("\nsentiment: ", sentiment_score)
        if sentiment_score >= 0.7:
            break 
        SYS_PROMPT = (
            "You are a helpful assistant. You will be given a Previous Answer and Feedback. "
            "Your job is to generate a REVISED ANSWER that addresses the feedback. "
            "IMPORTANT: Output ONLY the revised answer. Do not acknowledge the feedback. "
            "Do NOT start with 'Here is the revised plan'. Just output the content."
        )
        formatted_feedback = (
            f"Please revise the following answer based on the feedback provided:\n\n"
            f"ORIGINAL PROMPT: {prompt}\n"
            f"PREVIOUS ANSWER:\n{new_ans}\n\n"
            f"FEEDBACK:\n{feedback}\n\n"
            f"Provide a REVISED, complete answer now that addresses all points of the feedback."
        )
        prev = new_ans
        res = call_model_chat_completions(prompt=formatted_feedback, system=SYS_PROMPT, max_tokens=4096, temperature=temp)["text"]
        new_ans = res.strip() if res is not None else prev
    return new_ans

def assumption_explicit_reasoning(prompt: str, domain: str, temp: float = 0.0) -> str:
    init_ans = chain_of_thought(prompt, temp) #2
    extraction_sys_prompt = (
        f"You are a critical evaluator specializing in {domain}. our task is to read the provided draft answer and extract all implicit assumptions that must be true for that answer to hold. Return only assumptions. "
    "Each assumption must be explicit, minimal, falsifiable, and phrased as a concrete condition about the world, data, behavior, or model inputs. Do not justify or evaluate them."
    "Do not restate the draft answer."
    "List assumptions as numbered items.\n"
    "--- Example ---\n"
    "INPUT: The project will be completed in 5 days because all tasks are budgeted for 5 days of work.\n"
    "OUTPUT:\n"
    "1. The daily productivity of the team matches the estimated effort in the budget.\n"
    "2. No external factors will interfere with the planned schedule.\n"
    "3. The resources required for the tasks are all available immediately."
    )
    assumptions = call_model_chat_completions(prompt=init_ans, system=extraction_sys_prompt, max_tokens=1024)["text"] #3
    if not assumptions: #no assumptions found
        assumptions = "No specific assumptions found."
    reasoning_sys_prompt = (
        f"You are a {domain} expert. Use the following assumptions to construct your solution. "
        "Think step-by-step. At the end of your solution, write 'Final Answer:' followed by the final, concise result. "
        "Do not include any other commentary, and DO NOT list the assumptions separately. "
        f"Assumptions: {assumptions}"
    )
    reasoning_resp = call_model_chat_completions( #4
        prompt=prompt,                 
        system=reasoning_sys_prompt, 
        max_tokens=4096, 
        temperature=temp
    )["text"]
    isMath = (domain == "Math")
    final_answer = extract_final_answer(reasoning_resp, isMath=isMath)
    if len(final_answer) > 500 or final_answer == reasoning_resp: #i.e. extraction didnt work
        extract_answer_system_prompt = ( #5
            "Extract the complete final answer from this solution. "
            "For multiple choice questions, return ONLY the letter (A, B, C, D, or E). "
            "For numerical answers, return just the number. "
            "For plans or lists, extract all the steps. "
            "For reasoning questions, extract the conclusion. "
            "Reply with only the final answer itself—no explanations, no commentary.\n"
            "If you cannot determine a clear final answer, return the last meaningful statement.\n"
            "--- Examples ---\n"
            "INPUT 1: The total cost is $25, and the tax is $2.50. Final Answer: 27.50\n"
            "OUTPUT 1: 27.50\n"
            "INPUT 2: The required steps are: 1. Collect data. 2. Analyze. 3. Finalize. Final Answer: 1. Collect data. 2. Analyze. 3. Finalize\n"
            "OUTPUT 2: 1. Collect data. 2. Analyze. 3. Finalize"
        )
        answer = call_model_chat_completions(prompt=reasoning_resp, system=extract_answer_system_prompt, max_tokens=512, temperature=0.0)["text"]
        if answer and answer.strip():
            return answer.strip()
    return final_answer if final_answer else reasoning_resp.strip() if reasoning_resp is not None else ""