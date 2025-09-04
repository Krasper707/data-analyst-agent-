# app.py
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import openai
import json
import pandas as pd

# Import our agent's tools
import tools

# Initialize FastAPI app
app = FastAPI()

# Initialize the OpenAI client.
# It will automatically pick up credentials from Hugging Face Secrets.
client = openai.OpenAI()

# Give the tools module access to the initialized OpenAI client
tools.set_openai_client(client)

@app.get("/")
async def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Data Analyst Agent API is running!"}

@app.post("/api/")
async def analyze_data(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    files: List[UploadFile] = File([], alias="files"),
):
    """
    Main endpoint to handle data analysis tasks. It orchestrates scraping,
    data extraction, code generation, and code execution.
    """
    questions_text = (await questions_file.read()).decode("utf-8")
    
    # Simple router: Check if the task involves scraping a URL.
    if "scrape" in questions_text.lower() and "http" in questions_text.lower():
        
        # --- AGENT WORKFLOW ---

        # Step 1: PERCEIVE - Get the fully rendered HTML from the URL using Playwright
        print("Step 1: Fetching dynamic HTML from URL...")
        url = next((word for word in questions_text.split() if word.startswith("http")), None)
        if not url:
            return {"error": "Scraping task detected, but no URL was found."}
        
        html_content = await tools.get_dynamic_html(url)
        if isinstance(html_content, str) and "Error" in html_content:
            return {"error": html_content}

        # Step 2: DECIDE - Ask the LLM to identify the best table to use for the task
        print("Step 2: Asking LLM to choose the best table index...")
        choice_json_str = tools.choose_best_table_from_html(html_content, questions_text)
        try:
            choice = json.loads(choice_json_str)
            if "error" in choice:
                return {"error": choice["error"]}
            table_index = choice.get("index")
            if table_index is None or not isinstance(table_index, int):
                return {"error": "LLM failed to return a valid integer index for the table."}
        except (json.JSONDecodeError, TypeError):
            return {"error": f"Failed to decode LLM response for table choice: {choice_json_str}"}

        # Step 3: ACT (Extraction) - Extract the chosen table into a pandas DataFrame
        print(f"Step 3: Extracting table with index '{table_index}'...")
        df = tools.extract_table_to_dataframe(html_content, table_index)
        if isinstance(df, str):
            return {"error": df}

        # --- STEP 4: GENERATE & EXECUTE PYTHON CODE ---
        print("Step 4: Generating Python code for analysis...")

        # Prepare a concise summary of the DataFrame for the LLM prompt
        df_head = df.head().to_string()
        df_info = f"Here is the head of the pandas DataFrame, named 'df':\n{df_head}"

        system_prompt = """
        You are an AI data analyst. Your ONLY task is to write a Python script that operates on a pre-existing pandas DataFrame named `df`.

        **URGENT AND CRITICAL INSTRUCTIONS:**
        - The pandas DataFrame `df` is ALREADY in memory.
        - The pandas library is ALREADY imported as `pd`.
        - The regex library is ALREADY imported as `re`.
        - DO NOT include any `import` statements in your code.
        - DO NOT write any code to read or load data.
        - Your entire output must be ONLY the raw Python code. No markdown, no comments, no explanations.

        **Your script MUST:**
        1.  Perform data cleaning on the `df` DataFrame first.
        2.  For EACH question the user asks, you MUST `print()` the final answer.
        """
        user_prompt = f"{df_info}\n\nPlease write a Python script to answer the following questions:\n\n{questions_text}"

        try:
            # Generate the Python code using the LLM
            completion = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            response_content = completion.choices[0].message.content
            
            # Extract the code from the markdown block (e.g., ```python\n...\n```)
            python_code = response_content.strip().replace("```python", "").replace("```", "").strip()
            
            # Step 5: ACT (Execution) - Run the generated code using our tool
            print(f"--- Generated Python Code ---\n{python_code}\n-----------------------------")

            print("Step 5: Executing generated code.")
            execution_result = tools.run_python_code_on_dataframe(df, python_code)
            
            # The result is the captured print output. Format it into a JSON array of strings.
            final_answers = [line for line in execution_result.strip().split('\n') if line.strip()]
            
            return final_answers

        except Exception as e:
            return {"error": f"An error occurred during code generation or execution: {str(e)}"}

    else:
        # Handle non-scraping, general knowledge tasks
        return {"response": "This is a non-scraping task."}