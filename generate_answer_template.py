#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
from agent import run_agent
from concurrent.futures import ThreadPoolExecutor


#TODO: review # of API calls
#TODO: math answers returning in LaTeX
#TODO: 1/6 of outputs = "" - ok?

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")
MAX_WORKERS = 64 #concurrency setting


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data

def get_prediction(question: Dict[str, Any]) -> Dict[str, str]: #helper function to run the agent on a single question
    prompt = question["input"]
    domain = question.get("domain", "unknown") 
    prediction = run_agent(prompt, domain)
    return {"output": prediction}

def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]: #Iterates through questions CONCURRENTLY (faster)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor: #use ThreadPoolExecutor for concurrent processing
        results_iterator = executor.map(get_prediction, questions)
        answers = list(tqdm(
            results_iterator, 
            total=len(questions), 
            desc=f"Generating Answers (Concurrent, {MAX_WORKERS} Workers)"
        ))
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()