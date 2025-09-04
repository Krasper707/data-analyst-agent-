# app.py
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import os
import openai
import json

# Import our new tool
from tools import scrape_url_to_dataframe

app = FastAPI()

client = openai.OpenAI()

@app.get("/")
async def read_root():
    return {"message": "Data Analyst Agent API is running!"}

@app.post("/api/")
async def analyze_data(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    files: List[UploadFile] = File([], alias="files"),
):
    questions_text = (await questions_file.read()).decode("utf-8")

    # --- LLM Decides Which Tool to Use ---
    # We will use a more advanced agent framework later.
    # For now, a simple keyword check is enough to demonstrate the concept.
    
    if "scrape" in questions_text.lower() and "http" in questions_text.lower():
        # This is a scraping task. Let's find the URL.
        url = None
        for word in questions_text.split():
            if word.startswith("http"):
                url = word
                break
        
        if not url:
            return {"error": "Scraping task detected, but no URL found in the question."}

        # Call our scraping tool
        scraped_data = scrape_url_to_dataframe(url)

        # Check if the tool returned a DataFrame or an error string
        if isinstance(scraped_data, str):
            # The tool returned an error
            return {"error": scraped_data}
        
        # For now, just return the first 5 rows of the DataFrame as JSON
        # In the next step, the LLM will analyze this data.
        return {
            "status": "Scraping complete",
            "url": url,
            "data_preview": json.loads(scraped_data.head().to_json(orient="records"))
        }

    else:
        # This is a general knowledge task, same as before.
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": questions_text}
                ]
            )
            llm_response = completion.choices[0].message.content
            return {"status": "LLM query complete", "response": llm_response}
        except Exception as e:
            return {"error": f"Error calling LLM: {e}"}