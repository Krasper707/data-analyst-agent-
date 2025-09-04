# app.py
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import openai
import json
import pandas as pd

# Import our new set of tools
import tools

app = FastAPI()
client = openai.OpenAI()
tools.set_openai_client(client) # Give the tools module access to the client

@app.get("/")
async def read_root():
    return {"message": "Data Analyst Agent API is running!"}

@app.post("/api/")
async def analyze_data(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    files: List[UploadFile] = File([], alias="files"),
):
    questions_text = (await questions_file.read()).decode("utf-8")
    
    if "scrape" in questions_text.lower() and "http" in questions_text.lower():
        url = next((word for word in questions_text.split() if word.startswith("http")), None)
        if not url:
            return {"error": "Scraping task detected, but no URL was found."}

        # --- AGENT WORKFLOW ---
        # 1. PERCEIVE
        print(f"Step 1: Fetching dynamic HTML from {url}")
        html_content = await tools.get_dynamic_html(url)
        if "Error" in html_content:
            return {"error": html_content}

        # 2. DECIDE
        print("Step 2: Asking LLM to choose the best table index.")
        task_description = f"Find a table with the following information: {questions_text}"
        choice_json_str = tools.choose_best_table_from_html(html_content, task_description)
        
        try:
            choice = json.loads(choice_json_str)
            if "error" in choice:
                return {"error": choice["error"]}
            table_index = choice.get("index") # Get the index from the LLM response
            if table_index is None or not isinstance(table_index, int):
                return {"error": "LLM failed to return a valid integer index for the table."}
        except json.JSONDecodeError:
            return {"error": f"Failed to decode LLM response for table choice: {choice_json_str}"}

        # 3. ACT
        print(f"Step 3: Extracting table with index '{table_index}'.")
        df_or_error = tools.extract_table_to_dataframe(html_content, table_index) # Use the index
        if isinstance(df_or_error, str):
            return {"error": df_or_error}
        
        # 4. ANALYSIS (This part is unchanged)
        print("Step 4: Analyzing data with LLM.")
        data_string = df_or_error.to_csv(index=False)
        if len(data_string) > 15000:
            data_string = df_or_error.head(50).to_csv(index=False)
        
        system_prompt = """
        You are an expert data analyst agent... Respond with a JSON object: {\"answers\": [...]}
        """
        user_prompt = f"Data:\n{data_string}\n\nQuestions:\n{questions_text}"

        try:
            completion = client.chat.completions.create(model="gpt-4o", response_format={"type": "json_object"}, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
            response_data = json.loads(completion.choices[0].message.content)
            return response_data.get("answers", {"error": "LLM did not return answers in the expected format."})
        except Exception as e:
            return {"error": f"Error during final analysis: {str(e)}"}

    else:
        # Handle non-scraping tasks here
        return {"response": "This is a non-scraping task."}
