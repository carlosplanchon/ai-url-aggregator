# ai_url_aggregator

> **Note**: *This is a small experimental library, provided as-is.*

**ai_url_aggregator** is a Python tool that leverages **Perplexity** and **OpenAI** to search the internet for relevant URLs, normalize and deduplicate them, check their availability, and then select the most relevant ones based on GPT analysis.

---

## Features

1. **Search Across Models**  
   Queries Perplexity’s `sonar-reasoning-pro` model (configurable) several times in parallel. An OpenAI model extracts the URLs from each answer, and the pages Perplexity searched (`search_results`) are added to the list.
2. **Clean & Filter**  
   - Normalizes URLs: adds `https://` to bare domains such as `www.example.com`, lowercases the host, and drops trailing slashes and fragments.  
   - Removes duplicates, preferring `https://` over `http://` and treating `www.example.com` and `example.com` as the same site.
3. **Online Check**  
   - Verifies each URL’s availability (final status `200` or `403`) concurrently with `requests`, following redirects and falling back from `HEAD` to `GET`.  
   - Skips URLs, including redirect targets, that resolve to private, loopback or other non-public addresses.
4. **Relevance Ranking**  
   - Uses an OpenAI model to select the websites most relevant to your query, keeping only URLs from the checked list.

### DeepWiki Docs: [https://deepwiki.com/carlosplanchon/ai_url_aggregator](https://deepwiki.com/carlosplanchon/ai_url_aggregator)

---

## Installation

### 1. Install via PyPI

```bash
uv add ai_url_aggregator
```

### 2. Set Environment Variables

You must provide your **Perplexity** and **OpenAI** API keys:

```bash
export PERPLEXITY_API_KEY="PERPLEXITY_API_KEY"
export OPENAI_API_KEY="OPENAI_API_KEY"
```

Replace `"PERPLEXITY_API_KEY"` and `"OPENAI_API_KEY"` with your actual API keys. They are read the first time an API is called, so importing the package works without them. You can also pass your own clients instead (see [Configuration](#configuration)).

### 3. (Optional) Install from Source

1. **Clone or Download** this repository.  
2. **Install Dependencies**:
   ```bash
   uv sync
   ```
   This ensures all required libraries (like `openai`, `requests`, etc.) are installed, along with `pytest`.
3. **Run the Tests**:
   ```bash
   uv run pytest
   ```
   The tests use fake API clients and a local HTTP server, so they need neither API keys nor internet access.

---

## How It Works

1. **`query_models(query: str, ...) -> list[str]`**  
   - Sends a query to Perplexity (`sonar-reasoning-pro` by default) and removes the model’s `<think>` reasoning block from the answer.
   - Parses the answer with an OpenAI model into a structured list of URLs and adds the pages Perplexity searched.

2. **`keep_https(urls: list[str]) -> list[str]`**  
   - Normalizes the URLs (see `normalize_url`) and removes duplicates, preferring `https://` and ignoring `www.`.
   - Drops anything that isn’t an `http(s)` URL.

3. **`execute_query_multiple_times(query: str, num_runs: int = 5, ...) -> list[str]`**  
   - Runs the query multiple times in parallel to gather more URLs.
   - Deduplicates the normalized results, keeping the order in which they were found. Runs that hit an API error are logged and skipped; if every run fails, the error is raised.

4. **`check_urls_online(urls: list[str], ...) -> list[str]`**  
   - Checks the URLs concurrently and keeps the reachable ones (final status `200` or `403`).
   - Follows redirects, and tries `GET` when `HEAD` fails or returns another status, without downloading the body.
   - Treats URLs that resolve to non-public addresses as offline unless `allow_private_addresses=True`.

5. **`search_for_web_urls(query: str, num_runs: int = 5, ...) -> list[str]`**  
   - Brings all the above together:
     1. Executes a query multiple times.
     2. Prefers HTTPS versions of each site.
     3. Verifies URL reachability.
     4. Returns a final list of online, deduplicated URLs.

6. **`get_top_relevant_websites(website_urls: list[str], query: str | None = None, ...) -> list[Website]`**  
   - Uses an OpenAI model to select the most relevant websites for `query` from the final list of URLs.
   - Discards any URL the model returns that isn’t in the list.

### Configuration

`query_models`, `execute_query_multiple_times` and `search_for_web_urls` accept these keyword arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `perplexity_model` | `"sonar-reasoning-pro"` | Perplexity model used for the search. |
| `openai_model` | `"gpt-4o"` | OpenAI model that extracts the URLs. It must support Structured Outputs. |
| `include_search_results` | `True` | Also return the pages Perplexity searched. |
| `perplexity_client` | `None` | An `openai.OpenAI` client pointed at Perplexity’s API. |
| `openai_client` | `None` | An `openai.OpenAI` client. |

`get_top_relevant_websites` accepts `openai_model` and `openai_client`. Clients that aren’t passed are created from the environment variables:

```python
from openai import OpenAI
from ai_url_aggregator import PERPLEXITY_BASE_URL, search_for_web_urls

online_urls = search_for_web_urls(
    query="...",
    perplexity_client=OpenAI(api_key="...", base_url=PERPLEXITY_BASE_URL),
    openai_client=OpenAI(api_key="..."),
)
```

Progress is reported through the standard `logging` module, under the `ai_url_aggregator` logger. To see it:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

---

## Usage Example

Once installed and your environment variables are set, you can do:

```python
from pprint import pprint

from ai_url_aggregator import (
    search_for_web_urls,
    get_top_relevant_websites
)

# Example query:
query = "Give me a list of all the real estate agencies in Uruguay."

# Step 1: Get a cleaned, deduplicated, and verified list of URLs
online_urls = search_for_web_urls(query=query)

print("--- Online URLs ---")
pprint(online_urls)

# Step 2: Get the websites most relevant to the query from the final list
most_important_websites = get_top_relevant_websites(
    website_urls=online_urls,
    query=query
)

print("--- Most Important Websites ---")
pprint(most_important_websites)
```

Example output of step 2 (illustrative; results vary between runs):
```python
[Website(name='InfoCasas', url='https://www.infocasas.com.uy'),
 Website(name='CasasWeb', url='https://www.casasweb.com.uy'),
 Website(name='Mercado Libre Inmuebles', url='https://www.mercadolibre.com.uy/inmuebles'),
 Website(name='Uruguay Inmobiliarias', url='https://www.uruguayinmobiliarias.com')]
```

---

## License

This project is distributed under the **MIT License**. See `LICENSE` for more information.

---

All suggestions and improvements are welcome!
