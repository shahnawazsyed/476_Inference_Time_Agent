# Final Project: Inference-Time Agent

# Running
python generate_answer_template.py

# Files
- api.py – provided API interface
- strategies.py – reasoning methods
- agent.py – core logic selecting strategies
- generate_answer_template.py – run the full agent and output answers in JSON format

# Overview
This agent’s goal is to solve complex reasoning problems across multiple domains (math, common sense, planning, coding, future prediction) by selecting the best inference-time reasoning strategy. It consists of the following core components:
- API interface: provided LLM that provides answers for a given prompt and system instruction, subject to temperature and max tokens parameters
- Agent logic: Decisioning logic that calls a particular strategy based on the domain of the question
- Strategies: Chain of Thought Prompting (helper), Self Consistency, Self Refinement, Assumption Explicit Reasoning. These strategies frequently employ each other in their execution (see below)
The main execution flow is in generate_answer_template.py. For each prompt, it calls run_agent() in agent.py to extract an answer. After iterating over all prompts, it outputs the answers to a JSON file. 
# Agent routing and API architecture
The agent's core function is to map the problem's domain to a specific reasoning strategy. The mapping is defined below in the reasoning strategies section. The conditional logic is implemented in the run_agent(prompt, domain) function in agent.py.
All strategies rely on a single, standardized function to communicate with the underlying LLM. This is the call_model_chat_completions() function in api.py. The function handles system prompts, user prompts, temperature settings, and maximum token limits.
# Domain-Specific Reasoning Strategies (all located in strategies.py)
## Strategy 1: Self-Consistency - self_consistency(prompt, isMath, num_samples, verbose)
This strategy is best used for the "math" and "common_sense" domains. For the former, the prompt first undergoes a conversion step to ensure LaTeX is converted to plain text which is easier to read by the LLM (convertToPlainText(prompt) in strategies.py). Self consistency concurrently (using ThreadPoolExecutor) generates multiple independent Chain of Thought (CoT) samples (by default 7 samples) using random temperature values. It then selects the final answer based on the majority vote. In the process, it calls chain_of_thought(prompt, temp) in strategies.py multiple times.
## Strategy 2: Self-Refine - self_refine(prompt, domain, temp, max_iter, verbose)
This strategy is best used for the “planning” and “coding” domains. After first calling the API for an initial answer to the prompt, it utilizes an iterative process of gathering feedback via another LLM call, using a refine_sys_prompt that is specialized for the domain, and then combining this with the initial answer and prompt to create a new system prompt for a revised attempt. This process continues until a sentiment score above 0.7 is judged by the LLM or the maximum iteration limit is reached. 
## Strategy 3: Assumption-Explicit Reasoning - assumption_explicit_reasoning(prompt, domain, temp)
This strategy is best used for the "future prediction” domain. It enhances the standard CoT by extracting all implicit assumptions from an initial CoT response and then running the prompt through the model again with these in mind. This forces the LLM to address assumptions which may be unrealistic. 
## Helper Strategy: Chain of Thought (CoT) reasoning - chain_of_thought(prompt, temp)
While not directly used by the agent, CoT reasoning is used by nearly all other strategies. It works by providing an initial instruction to the model to provide a full, complete solution to the prompt while also detailing its thought process. This forces the LLM to "show its work" by outputting not just the final answer, but also the reasoning process used to reach it. The final answer is then extracted from the initial output, including the steps if the prompt is related to planning. 
