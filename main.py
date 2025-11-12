"""
main.py

Entry point for running the final project.

Responsibilities:
1. Load development or test data (JSON format).
2. Call the agent for each instance.
3. Save results (e.g., data/outputs.json).
4. Ensure efficient execution (<20 API calls per instance).
"""

import json


if __name__ == "__main__":
    main()

def main():
    input_path = "data/cse476_final_project_dev_data.json"
    with open(input_path, "r") as f:
        inputs = json.load(f)
    for input in inputs:
        pass
        #TODO: call agent.py with the prompt