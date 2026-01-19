import sys
sys.path.append('e:/DEVELOPMENT/PROJECTS/ACTIVE/SIMENGINE/backend')
from backend.app.engines.reasoner import ReasoningEngine

def run_manual_tests():
    engine = ReasoningEngine()

    test_cases = [
        {
            "description": "Valid JSON with a valid citation",
            "input": """
            {
                \"key\": \"value\"
            }
            [CITATION:valid_citation]
            """,
        },
        {
            "description": "Invalid citation format",
            "input": """
            {
                \"key\": \"value\"
            }
            [CITATION:invalid!citation]
            """,
        },
        {
            "description": "Missing citations",
            "input": """
            {
                \"key\": \"value\"
            }
            """,
        },
        {
            "description": "Empty input",
            "input": "",
        },
        {
            "description": "Large input",
            "input": "{" + ",".join([f"\\\"key{i}\\\": \\\"value{i}\\\"" for i in range(1000)]) + "}",
        }
    ]

    for test in test_cases:
        description = test["description"]
        input_data = test["input"]
        print(f"Running test: {description}")
        try:
            engine.load(input_data)
            engine.run()
            print("Test passed.")
        except Exception as e:
            print(f"Test failed with exception: {e}")