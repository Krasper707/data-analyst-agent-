# tools.py (Index-based Version)
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import json
import openai
import pandas as pd
import re
import io
import sys
from contextlib import redirect_stdout


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import base64




client = None
def set_openai_client(c):
    global client
    client = c

async def get_dynamic_html(url: str) -> str:
    # This function remains the same
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=20000, wait_until='networkidle')
            html_content = await page.content()
        except Exception as e:
            await browser.close()
            return f"Error fetching page with Playwright: {e}"
        await browser.close()
        return html_content

def choose_best_table_from_html(html_content: str, task_description: str) -> str:
    """
    Uses an LLM to identify the best table by its INDEX.
    Returns a JSON object with the table's index, e.g., {"index": 0}.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    tables = soup.find_all('table')

    if not tables:
        return '{"error": "No tables found on the page."}'

    table_summaries = []
    for i, table in enumerate(tables):
        rows = table.find_all('tr')
        sample_text = ""
        for row in rows[:3]:
            cells = row.find_all(['td', 'th'])
            sample_text += " | ".join(cell.get_text(strip=True) for cell in cells[:4]) + "\n"
        
        table_summaries.append({
            "index": i, # Use the index as the identifier
            "sample_data": sample_text.strip()
        })
    
    system_prompt = """
    You are an expert web scraping assistant. I will provide a list of tables, each identified by a numerical index.
    Based on the user's task, your job is to identify the single best table.
    Respond with a single JSON object containing the index of the best table, like this: {"index": 1}
    """
    user_prompt = f"User's task: '{task_description}'\n\nHere are the tables I found:\n{json.dumps(table_summaries, indent=2)}"

    try:
        completion = client.chat.completions.create(
            model="gpt-5-nano",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f'{{"error": "LLM error in choosing table: {str(e)}"}}'

def extract_table_to_dataframe(html_content: str, table_index: int) -> (pd.DataFrame | str):
    """Extracts a specific table from HTML using its index into a DataFrame."""
    soup = BeautifulSoup(html_content, 'lxml')
    tables = soup.find_all('table')
    
    if not 0 <= table_index < len(tables):
        return f"Error: Invalid table index {table_index}. Only {len(tables)} tables were found."

    selected_table = tables[table_index]
    
    try:
        df_list = pd.read_html(io.StringIO(str(selected_table)))
        if not df_list:
            return "Error: Pandas could not parse the selected table."
        return df_list[0]
    except Exception as e:
        return f"Error converting table to DataFrame: {e}"
    

def run_python_code_on_dataframe(df: pd.DataFrame, python_code: str) -> str:
    """
    Executes Python code with a DataFrame and common libraries available.
    Captures and returns any output printed to stdout.
    """
    output_stream = io.StringIO()
    
    # --- THIS IS THE CORRECTED SANDBOX SETUP ---
    # Create a single dictionary to serve as the global and local scope.
    # This ensures that all libraries are accessible everywhere inside the exec'd code.
    execution_scope = {
        'df': df,
        'pd': pd,
        're': re,
        'plt': plt,
        'sns': sns,
        'np': np,
        'LinearRegression': LinearRegression,
        'io': io,
        'base64': base64,
        '__builtins__': __builtins__ # Ensure basic built-ins are available
    }
    
    try:
        with redirect_stdout(output_stream):
            # Pass the scope dictionary as the 'globals' argument.
            # This makes 'pd', 're', etc. globally available to the script.
            exec(python_code, execution_scope)
        
        plt.close('all')
            
        result = output_stream.getvalue()
        if not result:
            return "Code executed successfully with no printed output."
        return result
        
    except Exception as e:
        plt.close('all')
        return f"Error executing code: {type(e).__name__}: {e}\n---\nCode that failed:\n{python_code}"
