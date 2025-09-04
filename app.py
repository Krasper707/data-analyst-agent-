# app.py
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import openai
import json
import pandas as pd
import re

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
    questions_text = (await questions_file.read()).decode("utf-8")
    
    if "scrape" in questions_text.lower() and "http" in questions_text.lower():
        # ... (Steps 1, 2, and 3 are the same: get html, choose table, extract df) ...
        url = next((word for word in questions_text.split() if word.startswith("http")), None)
        if not url: return {"error": "URL not found."}
        html_content = await tools.get_dynamic_html(url)
        if isinstance(html_content, str) and "Error" in html_content: return {"error": html_content}
        choice_json_str = tools.choose_best_table_from_html(html_content, questions_text)
        try:
            choice = json.loads(choice_json_str)
            table_index = choice.get("index")
            if table_index is None: return {"error": "LLM failed to choose table."}
        except: return {"error": "Failed to decode LLM table choice."}
        df = tools.extract_table_to_dataframe(html_content, table_index)
        if isinstance(df, str): return {"error": df}

        # --- STEP 4: GENERATE & EXECUTE PYTHON CODE ---
        print("Step 4: Generating Python code for analysis.")

        df_head = df.head().to_string()
        df_info = f"Here is the head of the pandas DataFrame, named 'df':\n{df_head}"

        # --- THIS IS THE CRITICAL FIX ---
        # Extract only the numbered questions to prevent the LLM from getting distracted.
        analysis_questions = re.findall(r"^\d+\.\s.*", questions_text, re.MULTILINE)
        cleaned_questions_text = "\n".join(analysis_questions)
        if not cleaned_questions_text:
             # Fallback if no numbered questions are found
            cleaned_questions_text = questions_text
        
        print(f"--- Cleaned Questions for Code Gen ---\n{cleaned_questions_text}\n--------------------------------------")
        # --- END OF FIX ---

        # Final, simplified system prompt
        system_prompt = """
        You are an expert Python data analyst. Your only job is to write a Python script.
        A pandas DataFrame `df` and libraries `pd`, `re`, `plt`, `sns`, `np`, `io`, `base64`, `LinearRegression` are pre-loaded.

        CRITICAL:
        - DO NOT import libraries.
        - DO NOT load data.
        - Your output must be ONLY raw Python code.
        - Clean the `df` DataFrame.
        - For each question, `print()` the answer.
        - For plots, print a base64 data URI.
        """
        
        user_prompt = f"{df_info}\n\nAnswer these questions with a Python script:\n\n{cleaned_questions_text}"

        try:
            completion = client.chat.completions.create(model="gpt-5-nano", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
            response_content = completion.choices[0].message.content
            python_code = response_content.strip().replace("```python", "").replace("```", "").strip()
            
            print(f"--- Generated Python Code ---\n{python_code}\n-----------------------------")
            
            print("Step 5: Executing generated code.")
            execution_result = tools.run_python_code_on_dataframe(df, python_code)
            
            final_answers = [line for line in execution_result.strip().split('\n') if line.strip()]
            return final_answers

        except Exception as e:
            return {"error": f"An error occurred during code generation or execution: {str(e)}"}

    else:
        return {"response": "This is a non-scraping task."}
