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
    Executes Python code with a DataFrame named 'df' available in the local scope.
    Captures and returns any output printed to stdout.
    """
    # Create a string stream to capture stdout
    output_stream = io.StringIO()
    
    # Create a local scope for the exec to run in, with 'df' pre-populated
    local_scope = {
        'df': df,
        'pd': pd,
        're': re
    }
    
    try:
        # Redirect stdout to our stream
        with redirect_stdout(output_stream):
            # Execute the code in the defined scope
            exec(python_code, {'__builtins__': __builtins__}, local_scope)
        
        # Get the captured output
        result = output_stream.getvalue()
        if not result:
            return "Code executed successfully with no printed output."
        return result
        
    except Exception as e:
        return f"Error executing code: {e}\n---\nCode that failed:\n{python_code}"
