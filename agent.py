import os
import re
import subprocess
import sys
import tempfile
from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv(".env.local")

client = OpenAI(
    api_key=os.getenv("kimi_api"),
    base_url="https://api.moonshot.ai/v1"
)
model = "kimi-k2.6"

def generate_code(task):
    """
    Generates Python code for the given task using the Kimi K2.6 model.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a Python coding assitant. "
                "Respond with only a Python code block and nothing else."
            },
            {
                "role": "user",
                "content": task
            }
        ],
        temperature=1,
    )
    raw = response.choices[0].message.content
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()

def fix_code(task, code, error):
    """
    Asks Kimi K2.6 to fix code that failed. Returns the corrected code.
    """
    prompt = (
        "The following Python code was written to solve this task:\n\n"
        f"TASK:\n{task}\n\n"
        f"CODE:\n{code}\n\n"
        "When executed, it produced this error:\n"
        f"{error}\n\n"
        "Fix the code and return only the corrected Python code block."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a Python coding assitant. "
                "Respond with only a Python code block and nothing else."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1,
    )
    raw = response.choices[0].message.content
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


def execute_code(code):
    """
    Executes the code in a temporary file and grabs the outputs and errors.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Error: code timed out after 10 seconds", -1
    finally:
        os.unlink(path)

def main():
    print("Coding Agent - Type a task, or 'quit' to exit.")
    while True:
        task = input("task> ").strip()
        if not task:
            continue
        if task.lower() in ("quit", "exit"):
            break
 
        code = generate_code(task)
        print(f"\n Generated Code \n{code}\n")
 
        stdout, stderr, returncode = execute_code(code)
 
        if returncode == 0:
            print(f"Output\n{stdout if stdout else '(no output)'}\n")
        else:
            print(f"Error\n{stderr}\n")

if __name__ == "__main__":
    main()
