# ai-url-aggregator

> **Note**: *This is a small experimental library, provided as-is.*

**ai-url-aggregator** is a Python tool that leverages **Perplexity** and **OpenAI** to search the internet for relevant URLs: it expands your query into several searches, runs them, normalizes, deduplicates and checks the results, and then classifies and ranks them with GPT.

[![Tests](https://github.com/carlosplanchon/ai-url-aggregator/actions/workflows/tests.yml/badge.svg)](https://github.com/carlosplanchon/ai-url-aggregator/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/ai-url-aggregator.svg)](https://pypi.org/project/ai-url-aggregator/)
[![Python versions](https://img.shields.io/pypi/pyversions/ai-url-aggregator.svg)](https://pypi.org/project/ai-url-aggregator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/carlosplanchon/ai-url-aggregator)

---

## Features

1. **Query Expansion**  
   An OpenAI model turns your query into several search queries, varying the wording, the language and the angle (for example, directories), to find more of the websites you’re after.
2. **Web Search**  
   Perplexity’s Search API runs those queries (up to 5 per request, each request billed once) and returns ranked pages with their title and an extract. Every URL comes from the search index, so none are made up.
3. **Clean & Filter**  
   - Normalizes URLs: adds `https://` to bare domains such as `www.example.com`, lowercases the host, and drops trailing slashes and fragments.  
   - Removes duplicates, preferring `https://` over `http://` and treating `www.example.com` and `example.com` as the same site.
4. **Online Check**  
   - Verifies each URL’s availability (final status `200` or `403`) concurrently with `requests`, following redirects and falling back from `HEAD` to `GET`.  
   - Skips URLs, including redirect targets, that resolve to private, loopback or other non-public addresses.
5. **Relevance Ranking**  
   - Uses an OpenAI model to pick the pages relevant to your query, judging each by its title and extract, classifying it (official site, directory, article or other) and ordering them from most to least relevant. Only URLs from the search results are kept.

---

## Installation

### 1. Install via PyPI

Requires Python 3.11 or later.

```bash
uv add ai-url-aggregator
```

The package is imported as `ai_url_aggregator`, with underscores.

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
   This ensures all required libraries (like `openai`, `perplexityai`, `requests`, etc.) are installed, along with `pytest`.
3. **Run the Tests**:
   ```bash
   uv run pytest
   ```
   The tests use fake API clients and a local HTTP server, so they need neither API keys nor internet access.

---

## How It Works

1. **`expand_query(query: str, num_queries: int = 5, ...) -> list[str]`**  
   - Returns `query` followed by up to `num_queries - 1` related queries written by an OpenAI model (`gpt-6-luna` with low reasoning effort by default).
   - If the model refuses or its output is cut off, it logs a warning and returns just `query`.

2. **`search_web(queries: Sequence[str], ...) -> list[SearchResult]`**  
   - Runs the queries through Perplexity’s Search API, 5 per request.
   - Returns a `SearchResult` with the `url`, `title` and `snippet` of each page found.

3. **`keep_https(urls: list[str]) -> list[str]`**  
   - Normalizes the URLs (see `normalize_url`) and removes duplicates, preferring `https://` and ignoring `www.`.
   - Drops anything that isn’t an `http(s)` URL.

4. **`check_urls_online(urls: list[str], ...) -> list[str]`**  
   - Checks the URLs concurrently and keeps the reachable ones (final status `200` or `403`).
   - Follows redirects, and tries `GET` when `HEAD` fails or returns another status, without downloading the body.
   - Treats URLs that resolve to non-public addresses as offline unless `allow_private_addresses=True`.

5. **`search_for_web_urls(query: str, num_queries: int = 5, ...) -> list[SearchResult]`**  
   - Brings all the above together:
     1. Expands the query.
     2. Searches the web.
     3. Normalizes and deduplicates the results, preferring HTTPS.
     4. Returns the results whose URLs are online.

6. **`get_top_relevant_websites(results: Sequence[SearchResult | str], query: str | None = None, ...) -> list[Website]`**  
   - Uses an OpenAI model to select the results relevant to `query`, judging each by its title and snippet, and returns them as `Website(name, url, kind)`, most relevant first. `kind` is `"official_site"`, `"directory"`, `"article"` or `"other"`.
   - Accepts plain URLs too, although the model then has only the URL to go on.
   - Discards any URL the model returns that isn’t in the list.
   - Returns an empty list, and logs a warning, if the model refuses or its output is cut off.

The OpenAI calls use the Responses API with `store=False`, so the responses aren’t stored for later retrieval.

### Configuration

`search_for_web_urls` accepts these keyword arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `num_queries` | `5` | Search queries to run, counting yours; the OpenAI model writes the rest. |
| `max_results` | `20` | Results per search query, the most Perplexity returns. |
| `country` | `None` | ISO 3166-1 alpha-2 code, such as `"UY"`, to favor results from that country. |
| `openai_model` | `"gpt-6-luna"` | OpenAI model that writes the queries. It must support Structured Outputs. |
| `openai_reasoning_effort` | `"low"` | Reasoning effort for the OpenAI calls. `None` leaves it out, for models without reasoning such as `gpt-4o`. |
| `openai_temperature` | `None` | Sampling temperature for the OpenAI calls. Reasoning models only accept it with `openai_reasoning_effort="none"`. `None` leaves it out. |
| `perplexity_client` | `None` | A `perplexity.Perplexity` client. |
| `openai_client` | `None` | An `openai.OpenAI` client. Its backend must support the Responses API. |

`expand_query` and `get_top_relevant_websites` accept the `openai_*` arguments, and `search_web` accepts `max_results`, `country` and `perplexity_client`.

For a model without reasoning, such as `gpt-4o`, leave the reasoning effort out and set a temperature instead:

```python
results = search_for_web_urls(
    query="...",
    openai_model="gpt-4o",
    openai_reasoning_effort=None,
    openai_temperature=0,
)
```

You can pass your own clients; any client you leave out is created from the environment variables:

```python
from openai import OpenAI
from perplexity import Perplexity
from ai_url_aggregator import search_for_web_urls

results = search_for_web_urls(
    query="...",
    perplexity_client=Perplexity(api_key="..."),
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

# Step 1: Search the web and keep the unique results that are online
results = search_for_web_urls(query=query, country="UY")

print("--- Online URLs ---")
pprint([result.url for result in results])

# Step 2: Get the websites relevant to the query, classified and ranked
most_important_websites = get_top_relevant_websites(results, query=query)

print("--- Most Important Websites ---")
pprint(most_important_websites)
```

Example output of step 2 (illustrative; results vary between runs):
```python
[Website(name='InfoCasas', url='https://www.infocasas.com.uy', kind='directory'),
 Website(name='CasasWeb', url='https://www.casasweb.com.uy', kind='directory'),
 Website(name='Mercado Libre Inmuebles', url='https://www.mercadolibre.com.uy/inmuebles', kind='directory'),
 Website(name='Uruguay Inmobiliarias', url='https://www.uruguayinmobiliarias.com', kind='directory')]
```

---

## License

This project is distributed under the **MIT License**. See `LICENSE` for more information.

---

All suggestions and improvements are welcome!
