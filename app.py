# app.py
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import os
import openai # Import the openai library

app = FastAPI()

# Initialize the OpenAI client.
# It will automatically pick up the OPENAI_API_KEY and OPENAI_BASE_URL
# from the environment variables (our Hugging Face Secrets).
client = openai.OpenAI()

@app.get("/")
async def read_root():
    return {"message": "Data Analyst Agent API is running!"}

@app.post("/api/")
async def analyze_data(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    files: List[UploadFile] = File([], alias="files"),
):
    # Read the content of questions.txt
    questions_content = await questions_file.read()
    questions_text = questions_content.decode("utf-8")

    # --- LLM INTEGRATION ---
    llm_response_content = "No response from LLM." # Default message
    try:
        # Create a simple prompt for the LLM
        completion = client.chat.completions.create(
            model="gpt-5-nano", # You can try other models like "mistralai/mistral-7b-instruct"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here are the questions I need answered:\n\n{questions_text}\n\nCan you acknowledge that you received them?"}
            ]
        )
        llm_response_content = completion.choices[0].message.content
    except Exception as e:
        # If the LLM call fails, we'll know why
        llm_response_content = f"Error calling LLM: {e}"
    # --- END LLM INTEGRATION ---

    # We will build a more structured response later.
    # For now, just return the raw LLM response.
    return {
        "status": "Processing complete",
        "received_questions": questions_text,
        "llm_acknowledgement": llm_response_content
    }