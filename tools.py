# tools.py
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import json
import openai

# Use the client initialized in the main app
client = None

def set_openai_client(c):
    global client
    client = c

async def get_dynamic_html(url: str) -> str:
    """Fetches the fully rendered HTML of a page using Playwright's ASYNC API."""
    # 'async with' is the asynchronous context manager
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
    Uses an LLM to identify the best table in the HTML for a given task.
    Returns a CSS selector for that table.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    tables = soup.find_all('table')

    if not tables:
        return '{"error": "No tables found on the page."}'

    table_summaries = []
    for i, table in enumerate(tables):
        # Create a unique, stable selector for each table
        selector = f"table_{i}"
        table['data-agent-selector'] = selector
        
        # Get a small sample of the table's text content
        rows = table.find_all('tr')
        sample_text = ""
        for row in rows[:3]:  # Sample first 3 rows
            cells = row.find_all(['td', 'th'])
            sample_text += " | ".join(cell.get_text(strip=True) for cell in cells[:4]) + "\n"
        
        table_summaries.append({
            "selector": selector,
            "sample_data": sample_text.strip()
        })
    
    system_prompt = """
    You are an expert web scraping assistant. I will provide a list of tables found on a webpage, each with a unique selector and a sample of its data.
    Based on the user's task, your job is to identify the single best table that contains the relevant information.
    Respond with a single JSON object containing the selector of the best table, like this: {"selector": "table_1"}
    """
    user_prompt = f"User's task: '{task_description}'\n\nHere are the tables I found:\n{json.dumps(table_summaries, indent=2)}"

    try:
        completion = client.chat.completions.create(
            model="gpt-5-nano",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        # We return the raw JSON string from the LLM
        return completion.choices[0].message.content
    except Exception as e:
        return f'{{"error": "LLM error in choosing table: {str(e)}"}}'

def extract_table_to_dataframe(html_content: str, selector: str) -> (pd.DataFrame | str):
    """Extracts a specific table from HTML using its selector into a DataFrame."""
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Find the table using our unique data attribute
    selected_table = soup.find('table', {'data-agent-selector': selector})
    
    if not selected_table:
        return f"Error: Could not find the table with selector '{selector}'."

    try:
        # We need to remove our custom attribute before pandas reads it
        del selected_table['data-agent-selector']
        df_list = pd.read_html(str(selected_table))
        if not df_list:
            return "Error: Pandas could not parse the selected table."
        return df_list[0]
    except Exception as e:
        return f"Error converting table to DataFrame: {e}"