"""
main.py

Entry point for running the final project.

Responsibilities:
1. Load development or test data (JSON format).
2. Call the agent for each instance.
3. Save results (e.g., data/outputs.json).
"""


#TODO: track # of api calls
#TODO: play around with temperature more (0.0 = deterministic)
#TODO: for some reason the coding is forgetting imports


from datetime import datetime
import json
from agent import run_agent
from tqdm import tqdm
import random
from evaluate import evaluate_outputs


def main():
    input_path = "data/cse476_final_project_dev_data.json"
    predictions_path = "outputs/predictions.jsonl"
    with open(input_path, "r") as f:
        all_inputs = json.load(f)
    SAMPLE_SIZE = 25
    random.seed(42)
    inputs = random.sample(all_inputs, SAMPLE_SIZE)
    predictions = []
    for i, input_item in tqdm(enumerate(inputs)):
        prompt = input_item["input"]
        domain = input_item["domain"]
        prediction = run_agent(prompt, domain)
        predictions.append({
            "id": input_item.get("id", i),
            "domain": input_item.get("domain", "unknown"),
            "input": prompt,
            "expected_output": input_item.get("output", ""),
            "prediction": prediction
        })
    with open(predictions_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"\nPredictions saved to {predictions_path}")
    eval_results = evaluate_outputs(
        test_data=inputs,
        predictions=[p["prediction"] for p in predictions]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output_path = f"outputs/{timestamp}_eval.json"
    with open(eval_output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to {eval_output_path}")


if __name__ == "__main__":
    main()