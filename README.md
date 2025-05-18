```markdown
# ai_url_aggregator

> **Note**: *This is a small experimental library, provided as-is.

**ai_url_aggregator** is a Python tool that leverages **Perplexity** and **OpenAI** to search the internet for relevant URLs, filter and deduplicate them, check their availability, and then select the most important ones based on GPT analysis.

---

## Features

1. **Search Across Models**  
   Uses Perplexity’s `sonar-reasoning` model to query the internet for URLs related to your prompt.
2. **Clean & Filter**  
   - Prefers `https://` links when both `http://` and `https://` are found for the same domain.  
   - Removes duplicates by collecting results into a `set`.
3. **Online Check**  
   - Verifies each URL’s availability (status codes `200` or `403`) using `requests`.
4. **Relevance Ranking**  
   - Uses an OpenAI model to select the most important websites from the deduplicated list of online URLs.

### DeepWiki Docs: [https://deepwiki.com/carlosplanchon/ai_url_aggregator](https://deepwiki.com/carlosplanchon/ai_url_aggregator)

---

## Installation

### 1. Install via PyPI

```bash
pip install ai_url_aggregator
```

### 2. Set Environment Variables

You must provide your **Perplexity** and **OpenAI** API keys:

```bash
export PERPLEXITY_API_KEY="PERPLEXITY_API_KEY"
export OPENAI_API_KEY="OPENAI_API_KEY"
```

Replace `"PERPLEXITY_API_KEY"` and `"OPENAI_API_KEY"` with your actual API keys.

### 3. (Optional) Install from Source

1. **Clone or Download** this repository.  
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This ensures all required libraries (like `openai`, `requests`, etc.) are installed.

---

## How It Works

1. **`query_models(query: str) -> list[str]`**  
   - Sends a query to Perplexity’s `sonar-reasoning` model.
   - Parses the Perplexity output with an OpenAI model into a structured list of URLs.

2. **`keep_https(urls: list[str]) -> list[str]`**  
   - Selects `https://` versions of URLs when duplicates exist, else keeps `http://`.

3. **`execute_query_multiple_times(query: str, num_runs: int) -> list[str]`**  
   - Runs the query multiple times to gather more URLs.
   - Deduplicates results using a `set`.

4. **`check_urls_online(urls: list[str]) -> list[str]`**  
   - Pings each URL to see if it’s reachable (status `200` or `403`).

5. **`search_for_web_urls(query: str, num_runs: int) -> list[str]`**  
   - Brings all the above together:
     1. Executes a query multiple times.
     2. Prefers HTTPS versions of each domain.
     3. Verifies URL reachability.
     4. Returns a final list of online, deduplicated URLs.

6. **`get_top_relevant_websites(website_urls: list[str]) -> list[Website]`**  
   - Uses an OpenAI model to select the most relevant (important) websites from the final list of URLs.

---

## Usage Example

Once installed and your environment variables are set, you can do:

```python
import prettyprinter
from ai_url_aggregator import (
    search_for_web_urls,
    get_top_relevant_websites
)

# Optional: install prettyprinter extras for nicer output
prettyprinter.install_extras()

# Example query:
query = "Give me a list of all the real state agencies in Uruguay."

# Step 1: Get a cleaned, deduplicated, and verified list of URLs
online_urls = search_for_web_urls(query=query)

print("--- Online URLs ---")
prettyprinter.cpprint(online_urls)

# Step 2: Get the most important websites from the final list
most_important_websites = get_top_relevant_websites(website_urls=online_urls)

print("--- Most Important Websites ---")
prettyprinter.cpprint(most_important_websites)
```

---

## License

This project is distributed under the **MIT License**. See `LICENSE` for more information.

---

All suggestions and improvements are welcome!
