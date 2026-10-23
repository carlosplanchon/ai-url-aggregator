"""Search the web for URLs with Perplexity and filter them with OpenAI."""

import ipaddress
import json
import logging
import socket
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache, partial
from typing import Literal, TypeVar
from urllib.parse import urlsplit, urlunsplit

import requests
from openai import OpenAI
from perplexity import Perplexity
from pydantic import BaseModel, Field, ValidationError
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_REASONING_EFFORT = "low"

# Limits of Perplexity's Search API.
MAX_QUERIES_PER_REQUEST = 5
MAX_RESULTS_PER_QUERY = 20

# Page text asked for per result, and sent to the model when ranking: enough
# to judge a page, little enough to rank many.
_PAGE_TOKENS = 128
_SNIPPET_CHARS = 500

# Final statuses that count as online; 403 usually means bot protection.
ONLINE_STATUS_CODES = frozenset({200, 403})

# Defaults of the online check: seconds per request, URLs checked at once.
DEFAULT_TIMEOUT = 5
DEFAULT_MAX_WORKERS = 10


@cache
def _default_perplexity_client() -> Perplexity:
    # Reads PERPLEXITY_API_KEY and raises PerplexityError if it is not set.
    return Perplexity()


@cache
def _default_openai_client() -> OpenAI:
    # The SDK reads OPENAI_API_KEY.
    return OpenAI()


class SearchResult(BaseModel):
    """A web page found by the search, with its title and an extract."""

    url: str
    title: str = ""
    snippet: str = ""


class QueryList(BaseModel):
    queries: list[str]


class Website(BaseModel):
    name: str = Field(description="The name of the website.")
    url: str = Field(description="The url of the website.")
    kind: Literal["official_site", "directory", "article", "other"] = Field(
        description=(
            "official_site: the site of the entity itself; directory: a "
            "listing of many entities; article: news or reference content."
        )
    )


class WebsiteList(BaseModel):
    websites_list: list[Website] = Field(
        description="The relevant websites, most relevant first."
    )


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _refusal(response) -> str | None:
    """The model's refusal message in a Responses API result, if any."""
    refusals = [
        content.refusal
        for item in response.output
        if item.type == "message"
        for content in item.content
        if content.type == "refusal"
    ]
    return " ".join(refusals) or None


def _model_options(
    temperature: float | None, reasoning_effort: str | None
) -> dict[str, float | dict[str, str]]:
    """Optional responses.parse() kwargs, leaving out the ones that are None."""
    options: dict[str, float | dict[str, str]] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if reasoning_effort is not None:
        options["reasoning"] = {"effort": reasoning_effort}
    return options


def _parse_structured(
    openai_client: OpenAI,
    *,
    model: str,
    instructions: str,
    user_input: str,
    text_format: type[_ModelT],
    temperature: float | None,
    reasoning_effort: str | None,
) -> _ModelT | None:
    """Ask an OpenAI model for a `text_format` object via the Responses API.

    Returns None, with a warning, if the model refuses or its output is cut
    off or invalid.
    """
    try:
        response = openai_client.responses.parse(
            model=model,
            instructions=instructions,
            input=user_input,
            text_format=text_format,
            # The responses are never retrieved, so OpenAI needn't keep them.
            store=False,
            **_model_options(temperature, reasoning_effort),
        )
    except ValidationError as exc:
        # The SDK raises this, not an OpenAIError, for truncated or invalid
        # JSON, e.g. when the output hits the token limit.
        logger.warning("OpenAI returned invalid structured output: %s", exc)
        return None
    parsed = response.output_parsed
    if parsed is None:
        logger.warning(
            "OpenAI returned no structured output "
            "(status: %s, reason: %s, refusal: %r)",
            response.status,
            getattr(response.incomplete_details, "reason", None),
            _refusal(response),
        )
    return parsed


def expand_query(
    query: str,
    num_queries: int = MAX_QUERIES_PER_REQUEST,
    *,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    openai_temperature: float | None = None,
    openai_client: OpenAI | None = None,
) -> list[str]:
    """Return `query` followed by up to `num_queries - 1` related queries.

    An OpenAI model writes the extra queries, varying the wording, language
    and angle to find more of the websites `query` asks for. If it refuses
    or its output is cut off, that is logged and only `query` is returned.
    """
    if num_queries < 1:
        raise ValueError("num_queries must be at least 1")
    queries = [query]
    if num_queries > 1:
        if openai_client is None:
            openai_client = _default_openai_client()
        query_list = _parse_structured(
            openai_client,
            model=openai_model,
            instructions=(
                f"Write {num_queries - 1} web search queries that, together "
                "with the user's request, find as many of the websites it asks "
                "for as possible. Vary the wording, use the local language of "
                "any place involved, and include queries for directories or "
                "listings of them."
            ),
            user_input=query,
            text_format=QueryList,
            temperature=openai_temperature,
            reasoning_effort=openai_reasoning_effort,
        )
        if query_list is not None:
            queries.extend(query_list.queries)
    # A dict drops repeats while keeping the order.
    unique = dict.fromkeys(q.strip() for q in queries if q.strip())
    return list(unique)[:num_queries]


def search_web(
    queries: Sequence[str],
    *,
    max_results: int = MAX_RESULTS_PER_QUERY,
    country: str | None = None,
    perplexity_client: Perplexity | None = None,
) -> list[SearchResult]:
    """Run `queries` through Perplexity's Search API and return the results.

    Queries go MAX_QUERIES_PER_REQUEST to a request, and each request is
    billed once. `country`, an ISO 3166-1 alpha-2 code such as "UY", favors
    results from that country.
    """
    if perplexity_client is None:
        perplexity_client = _default_perplexity_client()
    options = {} if country is None else {"country": country}
    results: list[SearchResult] = []
    for start in range(0, len(queries), MAX_QUERIES_PER_REQUEST):
        batch = list(queries[start : start + MAX_QUERIES_PER_REQUEST])
        response = perplexity_client.search.create(
            query=batch,
            max_results=max_results,
            # Without a total budget, Perplexity applies its default one,
            # which can be too small to give every page its share.
            max_tokens=len(batch) * max_results * _PAGE_TOKENS,
            max_tokens_per_page=_PAGE_TOKENS,
            **options,
        )
        results.extend(
            SearchResult(url=result.url, title=result.title, snippet=result.snippet)
            for result in response.results
        )
    return results


def normalize_url(url: str) -> str | None:
    """Return `url` in canonical form, or None if it is not an http(s) URL.

    Scheme-less URLs such as "www.example.com" get https://. The host is
    lowercased; default ports, trailing slashes and fragments are dropped.
    URLs with embedded credentials are rejected.
    """
    url = url.strip()
    if "://" not in url:
        url = f"https://{url}"
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if (
        parts.scheme not in ("http", "https")
        or not host
        or any(char.isspace() for char in host)
        or parts.username is not None
    ):
        return None
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    if port is not None and port != (443 if parts.scheme == "https" else 80):
        host = f"{host}:{port}"
    return urlunsplit(
        (parts.scheme, host, parts.path.rstrip("/"), parts.query, "")
    )


def _dedup_key(normalized_url: str) -> str:
    """Key shared by the http/https and www/non-www variants of a URL."""
    parts = urlsplit(normalized_url)
    host = parts.netloc.removeprefix("www.")
    return urlunsplit(("", host, parts.path, parts.query, ""))


def _unique_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Normalize and deduplicate results the way keep_https does URLs."""
    kept: dict[str, SearchResult] = {}
    for result in results:
        normalized = normalize_url(result.url)
        if normalized is None:
            logger.debug("Discarding non-web URL: %r", result.url)
            continue
        key = _dedup_key(normalized)
        if key not in kept or (
            kept[key].url.startswith("http://")
            and normalized.startswith("https://")
        ):
            kept[key] = result.model_copy(update={"url": normalized})
    return list(kept.values())


def keep_https(urls: list[str]) -> list[str]:
    """Normalize and deduplicate URLs, preferring https:// over http://.

    "www.example.com" and "example.com" count as the same site. Strings that
    are not http(s) URLs are dropped. The order of first appearance is kept.
    """
    results = _unique_results(SearchResult(url=url) for url in urls)
    return [result.url for result in results]


def _is_global_ip(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


def _is_public_url(url: str) -> bool:
    """Whether every address the URL's host resolves to is public."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False
    if not host:
        return False
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return False
    return bool(addresses) and all(
        _is_global_ip(address[4][0]) for address in addresses
    )


class _PublicAddressAdapter(HTTPAdapter):
    """Refuses to send requests to non-public addresses, redirects included.

    The host is resolved here and again when connecting, so this guards
    against SSRF on a best-effort basis; it does not stop DNS rebinding.
    """

    def send(self, request, *args, **kwargs):
        if not _is_public_url(request.url):
            raise requests.ConnectionError(
                f"Refusing to connect to a non-public address: {request.url}",
                request=request,
            )
        return super().send(request, *args, **kwargs)


def _is_online(url: str, timeout: float, allow_private_addresses: bool) -> bool:
    with requests.Session() as session:
        if not allow_private_addresses:
            adapter = _PublicAddressAdapter()
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        # Some servers reject or mishandle HEAD, so GET gets a turn too.
        for method in ("HEAD", "GET"):
            try:
                # stream=True downloads the headers but not the body.
                with session.request(
                    method,
                    url,
                    timeout=timeout,
                    allow_redirects=True,
                    stream=True,
                ) as response:
                    if response.status_code in ONLINE_STATUS_CODES:
                        return True
                    logger.debug(
                        "%s %s -> %d", method, url, response.status_code
                    )
            except (requests.RequestException, ValueError) as exc:
                logger.debug("%s %s failed: %s", method, url, exc)
    return False


def check_urls_online(
    urls: list[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    allow_private_addresses: bool = False,
) -> list[str]:
    """Return the URLs whose final status is in ONLINE_STATUS_CODES.

    URLs are checked concurrently. Redirects are followed, and GET is tried
    when HEAD fails or returns another status. URLs whose host or redirect
    targets resolve to private, loopback or other non-public addresses count
    as offline unless `allow_private_addresses` is True.
    """
    if not urls:
        return []
    check = partial(
        _is_online,
        timeout=timeout,
        allow_private_addresses=allow_private_addresses,
    )
    workers = min(max_workers, len(urls))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        online = list(executor.map(check, urls))
    return [url for url, is_online in zip(urls, online) if is_online]


def search_for_web_urls(
    query: str,
    num_queries: int = MAX_QUERIES_PER_REQUEST,
    *,
    max_results: int = MAX_RESULTS_PER_QUERY,
    country: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    openai_temperature: float | None = None,
    perplexity_client: Perplexity | None = None,
    openai_client: OpenAI | None = None,
) -> list[SearchResult]:
    """Search the web for `query` and return the online, unique results.

    `query` is expanded into `num_queries` queries (see expand_query), which
    Perplexity's Search API runs; the results are then normalized,
    deduplicated and checked online. `timeout` and `max_workers` configure
    that check (see check_urls_online).
    """
    queries = expand_query(
        query,
        num_queries,
        openai_model=openai_model,
        openai_reasoning_effort=openai_reasoning_effort,
        openai_temperature=openai_temperature,
        openai_client=openai_client,
    )
    logger.info("Searching with %d queries: %s", len(queries), queries)
    results = _unique_results(
        search_web(
            queries,
            max_results=max_results,
            country=country,
            perplexity_client=perplexity_client,
        )
    )
    online = set(
        check_urls_online(
            [result.url for result in results],
            timeout=timeout,
            max_workers=max_workers,
        )
    )
    logger.info("%d of %d unique results are online", len(online), len(results))
    return [result for result in results if result.url in online]


def get_top_relevant_websites(
    results: Sequence[SearchResult | str],
    query: str | None = None,
    *,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    openai_temperature: float | None = None,
    openai_client: OpenAI | None = None,
) -> list[Website]:
    """Ask an OpenAI model which `results` are relevant, most relevant first.

    The model judges each page by its title and snippet and classifies it.
    Pass the original `query` so it knows what to rank for. Plain URLs are
    accepted too. Only websites from `results` are returned, with the URL as
    given there. If the model refuses or its output is cut off, that is
    logged and the result is empty.
    """
    candidates = [
        result if isinstance(result, SearchResult) else SearchResult(url=result)
        for result in results
    ]
    if not candidates:
        return []
    if openai_client is None:
        openai_client = _default_openai_client()

    task = "Select the most important websites from this JSON list."
    if query:
        task = (
            "Select the websites from this JSON list that are relevant to "
            f"the following query.\nQuery: {query}"
        )
    pages = [
        {
            "url": candidate.url,
            "title": candidate.title,
            "snippet": candidate.snippet[:_SNIPPET_CHARS],
        }
        for candidate in candidates
    ]
    website_list = _parse_structured(
        openai_client,
        model=openai_model,
        instructions=(
            f"{task}\nJudge each one by its title and snippet, classify it, "
            "and order them from most to least relevant. Use only URLs from "
            "the list."
        ),
        user_input=json.dumps(pages, ensure_ascii=False),
        text_format=WebsiteList,
        temperature=openai_temperature,
        reasoning_effort=openai_reasoning_effort,
    )
    if website_list is None:
        return []

    # Map the model's URLs back to the input ones, dropping any made up.
    originals: dict[str, str] = {}
    for candidate in candidates:
        normalized = normalize_url(candidate.url)
        if normalized is not None:
            originals.setdefault(_dedup_key(normalized), candidate.url)
    selected: dict[str, Website] = {}
    for website in website_list.websites_list:
        normalized = normalize_url(website.url)
        original = originals.get(_dedup_key(normalized)) if normalized else None
        if original is None:
            logger.info("Dropping a URL that is not in the list: %s", website.url)
        elif original not in selected:
            selected[original] = website.model_copy(update={"url": original})
    return list(selected.values())
