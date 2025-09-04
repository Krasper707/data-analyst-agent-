# app.py
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import os

app = FastAPI()

# A simple root endpoint to confirm the app is running
@app.get("/")
async def read_root():
    return {"message": "Data Analyst Agent API is running!"}

# Our main API endpoint for data analysis tasks
@app.post("/api/")
async def analyze_data(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    files: List[UploadFile] = File([], alias="files"), # This will catch other files if sent
):
    # Read the content of questions.txt
    questions_content = await questions_file.read()
    questions_text = questions_content.decode("utf-8")

    response_messages = [f"Received questions:\n{questions_text}"]

    # Process other uploaded files
    for file in files:
        # You would typically save these to a temporary location
        # For now, just acknowledge receipt
        response_messages.append(f"Received file: {file.filename} (Content-Type: {file.content_type})")
        # Example: Save to a temporary file
        # with open(f"/tmp/{file.filename}", "wb") as f:
        #     f.write(await file.read())
        # response_messages.append(f"Saved {file.filename} to /tmp/")


    # This is where the core logic will go. For now, it's a placeholder.
    # The LLM will process questions_text and use other files.
    
    return {"status": "Processing initiated", "details": response_messages}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860) # Hugging Face Spaces typically use port 7860