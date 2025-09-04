
# Data Analyst Agent API

 <!-- Optional: Create a banner image for your project -->

An intelligent API that leverages Large Language Models (LLMs) to function as an autonomous data analyst. This agent can source data from the web or uploaded files, prepare and clean it, perform complex analysis and calculations, and generate visualizations on demand.

This project demonstrates an advanced **Planner-Executor** agent architecture, where an LLM first breaks down a complex task into a structured plan, and then a series of tools execute that plan to achieve the final result.

---

## 🚀 Key Features

-   **Multi-Step Task Planning:** Dynamically creates and executes multi-step plans to handle complex data analysis requests.
-   **Dynamic Web Scraping:** Uses Playwright to render JavaScript-heavy websites and an LLM to intelligently identify and extract the correct data tables.
-   **Code Interpreter:** Generates and executes Python code in a sandboxed environment for reliable and precise data cleaning, analysis, and statistical calculations using `pandas` and `scikit-learn`.
-   **Dynamic Visualization:** Creates plots and charts on the fly using `matplotlib` and `seaborn`, returning them as base64 data URIs.
-   **Multi-Source Data Handling:** Capable of processing data from web URLs, uploaded files (`.csv`, `.pdf`, etc.), and cloud storage (e.g., S3). <!-- Update this as you add more features -->
-   **API-First Design:** Exposes a simple yet powerful API endpoint to receive tasks and return results.

---

## 🛠️ Tech Stack & Architecture

This project is built with a modern, robust tech stack designed for building AI-powered applications.

-   **Backend:** FastAPI (Python)
-   **LLM Orchestration:** Custom Planner-Executor loop with OpenAI's `gpt-4o`
-   **Data Handling:** Pandas, NumPy
-   **Web Scraping:** Playwright (for dynamic sites), BeautifulSoup4 (for parsing)
-   **Visualization:** Matplotlib, Seaborn
-   **Machine Learning:** Scikit-learn
-   **Deployment:** Docker, Hugging Face Spaces

### Architecture: Planner-Executor Model

 <!-- Optional: A simple diagram showing the flow -->

1.  **Request Input:** The API receives a natural language task (e.g., "Scrape this URL, join with this CSV, and plot the results") and optional file attachments.
2.  **Planner Agent:** An LLM call analyzes the request and breaks it down into a structured JSON plan. For example: `[{"tool": "scrape_web"}, {"tool": "read_csv"}, {"tool": "run_python_code"}]`.
3.  **Executor Loop:** The Python backend iterates through the plan, calling the appropriate tool for each step.
4.  **Tool Execution:** Each tool (e.g., `scrape_web`, `run_python_code`) performs its specific task, storing its results in a shared context.
5.  **Code Interpreter:** The `run_python_code` tool asks the LLM to write Python code to perform the final analysis, which is then executed in a secure sandbox.
6.  **Response Output:** The final result, which can be a JSON array of text, numbers, or base64-encoded images, is returned to the user.

---

## 🏁 Getting Started & Usage

The agent is exposed via a single API endpoint. You can interact with it using any HTTP client, like `curl`.

**API Endpoint:** `https://karthix1-data-analyst-agent.hf.space/api/`

### Example Usage: Web Scraping, Analysis, and Visualization

This example asks the agent to scrape a Wikipedia page, answer several analytical questions, and generate a plot.

1.  **Create a `questions.txt` file:**
    ```
    Scrape the list of highest grossing films from Wikipedia. It is at the URL:
    https://en.wikipedia.org/wiki/List_of_highest-grossing_films

    Answer the following questions:
    1. How many $2 bn movies were released before 2000?
    2. Which is the earliest film that grossed over $1.5 bn?
    3. What's the correlation between the Rank and Peak?
    4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
       Return as a base-64 encoded data URI.
    ```

2.  **Send the request using `curl`:**
    ```bash
    curl -X POST "https://karthix1-data-analyst-agent.hf.space/api/" \
         -F "questions.txt=@questions.txt"
    ```

3.  **Expected Response:**
    A JSON array containing the answers to the four questions. The final answer will be a long data URI string representing the generated plot.
    ```json
    [
      "Answer 1: 1",
      "Answer 2: Titanic (1997)",
      "Answer 3: Correlation: 0.5389",
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA... (and so on)"
    ]
    ```

---

## 🔧 Running Locally

To run this project on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Karthix1/data-analyst-agent.git
    cd data-analyst-agent
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_or_aipipe_token"
    OPENAI_BASE_URL="optional_base_url_if_using_a_proxy"
    ```

3.  **Build and run with Docker (Recommended):**
    This ensures all dependencies, including Playwright's browsers, are correctly installed.
    ```bash
    docker build -t data-analyst-agent .
    docker run -p 8000:7860 -v $(pwd):/app --env-file .env data-analyst-agent
    ```
    The API will be available at `http://localhost:8000`.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
