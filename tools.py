# tools.py
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_url_to_dataframe(url: str) -> (pd.DataFrame | str):
    """
    Scrapes a given URL for the first HTML table and returns it as a pandas DataFrame.
    If no table is found or an error occurs, it returns an error message string.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'lxml')
        
        # Find the first table in the HTML. Wikipedia pages often have the main data here.
        table = soup.find('table', {'class': 'wikitable'})
        
        if not table:
            return "Error: No table with class 'wikitable' found on the page."

        # Use pandas to read the HTML table directly into a DataFrame
        # read_html returns a list of DataFrames, we want the first one.
        df_list = pd.read_html(str(table))
        if not df_list:
            return "Error: Pandas could not parse any tables from the HTML."
            
        df = df_list[0]
        return df

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    

# from playwright.sync_api import sync_playwright

# def scrape_dynamic_url(url: str) -> str:
#     """Scrapes a dynamic URL using Playwright and returns the final HTML."""
#     with sync_playwright() as p:
#         browser = p.chromium.launch()
#         page = browser.new_page()
#         page.goto(url, wait_until='networkidle') # Wait for network activity to cease
#         html_content = page.content()
#         browser.close()
#         return html_content
