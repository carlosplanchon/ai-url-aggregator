"""Search the web for URLs with Perplexity and filter them with OpenAI."""

import ipaddress
import json
import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import cache, partial
from urllib.parse import urlsplit, urlunsplit

import requests
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
DEFAULT_PERPLEXITY_MODEL = "sonar-reasoning-pro"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_REASONING_EFFORT = "low"

# Final statuses that count as online; 403 usually means bot protection.
ONLINE_STATUS_CODES = frozenset({200, 403})
MAX_PARALLEL_QUERIES = 5


@cache
def _default_perplexity_client() -> OpenAI:
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        # Given no key, the SDK would fall back to OPENAI_API_KEY and send it
        # to Perplexity.
        raise RuntimeError(
            "PERPLEXITY_API_KEY is not set; export it or pass perplexity_client."
        )
    return OpenAI(api_key=api_key, base_url=PERPLEXITY_BASE_URL)


@cache
def _default_openai_client() -> OpenAI:
    # The SDK reads OPENAI_API_KEY.
    return OpenAI()


class UrlList(BaseModel):
    urls: list[str]


def _strip_reasoning(text: str) -> str:
    """Remove the <think>...</think> block that reasoning models prepend."""
    _, marker, answer = text.rpartition("</think>")
    if marker:
        return answer.strip()
    # A truncated response can end inside the block, before any answer.
    return text.partition("<think>")[0].strip()


def _search_result_urls(response) -> list[str]:
    """URLs of the pages Perplexity searched, from its search_results field."""
    urls = []
    for result in getattr(response, "search_results", None) or []:
        if isinstance(result, dict):
            url = result.get("url")
        else:
            url = getattr(result, "url", None)
        if isinstance(url, str):
            urls.append(url)
    return urls


def _parsed_or_none(completion):
    """The structured output of a parse() call, or None if the model refused."""
    message = completion.choices[0].message
    if message.parsed is None:
        logger.warning(
            "OpenAI returned no structured output (refusal: %r)",
            message.refusal,
        )
    return message.parsed


def _model_options(
    temperature: float | None, reasoning_effort: str | None
) -> dict[str, float | str]:
    """Optional parse() kwargs, leaving out the ones that are None."""
    options = {"temperature": temperature, "reasoning_effort": reasoning_effort}
    return {name: value for name, value in options.items() if value is not None}


def query_models(
    query: str,
    *,
    perplexity_model: str = DEFAULT_PERPLEXITY_MODEL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    openai_temperature: float | None = None,
    include_search_results: bool = True,
    perplexity_client: OpenAI | None = None,
    openai_client: OpenAI | None = None,
) -> list[str]:
    """Ask Perplexity about `query` and return the URLs it comes up with.

    An OpenAI model extracts the URLs mentioned in the answer. With
    `include_search_results`, the pages Perplexity searched are added too.
    For an OpenAI model without reasoning, such as gpt-4o, pass
    `openai_reasoning_effort=None` (and optionally `openai_temperature=0`).
    """
    if perplexity_client is None:
        perplexity_client = _default_perplexity_client()
    if openai_client is None:
        openai_client = _default_openai_client()

    response = perplexity_client.chat.completions.create(
        model=perplexity_model,
        messages=[{"role": "user", "content": query}],
    )
    answer = _strip_reasoning(response.choices[0].message.content or "")
    logger.debug("Perplexity answer:\n%s", answer)

    urls: list[str] = []
    if answer:
        completion = openai_client.chat.completions.parse(
            model=openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract every website URL mentioned in the text. "
                        "Copy them exactly and do not invent any."
                    ),
                },
                {"role": "user", "content": answer},
            ],
            response_format=UrlList,
            **_model_options(openai_temperature, openai_reasoning_effort),
        )
        url_list = _parsed_or_none(completion)
        if url_list is not None:
            urls.extend(url_list.urls)

    if include_search_results:
        urls.extend(_search_result_urls(response))
    return urls


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


def keep_https(urls: list[str]) -> list[str]:
    """Normalize and deduplicate URLs, preferring https:// over http://.

    "www.example.com" and "example.com" count as the same site. Strings that
    are not http(s) URLs are dropped. The order of first appearance is kept.
    """
    kept: dict[str, str] = {}
    for url in urls:
        normalized = normalize_url(url)
        if normalized is None:
            logger.debug("Discarding non-web URL: %r", url)
            continue
        key = _dedup_key(normalized)
        if key not in kept or (
            kept[key].startswith("http://")
            and normalized.startswith("https://")
        ):
            kept[key] = normalized
    return list(kept.values())


def execute_query_multiple_times(
    query: str,
    num_runs: int = 5,
    *,
    perplexity_model: str = DEFAULT_PERPLEXITY_MODEL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    openai_temperature: float | None = None,
    include_search_results: bool = True,
    perplexity_client: OpenAI | None = None,
    openai_client: OpenAI | None = None,
) -> list[str]:
    """Run `query_models` `num_runs` times in parallel and merge the URLs.

    URLs are normalized and deduplicated in the order they were found.
    Runs that hit an API error are logged and skipped; if every run fails,
    the first error is raised.
    """
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    # Resolve the clients once, so a missing API key fails fast.
    if perplexity_client is None:
        perplexity_client = _default_perplexity_client()
    if openai_client is None:
        openai_client = _default_openai_client()
    run_query = partial(
        query_models,
        query,
        perplexity_model=perplexity_model,
        openai_model=openai_model,
        openai_reasoning_effort=openai_reasoning_effort,
        openai_temperature=openai_temperature,
        include_search_results=include_search_results,
        perplexity_client=perplexity_client,
        openai_client=openai_client,
    )

    logger.info("Running the query %d times", num_runs)
    workers = min(num_runs, MAX_PARALLEL_QUERIES)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_query) for _ in range(num_runs)]

    results: list[list[str]] = []
    errors: list[OpenAIError] = []
    for future in futures:
        try:
            results.append(future.result())
        except OpenAIError as exc:
            errors.append(exc)
    if not results:
        raise errors[0]
    for exc in errors:
        logger.warning("A query run failed and was skipped: %s", exc)

    # A dict keeps the order in which the URLs were first found.
    unique_urls = dict.fromkeys(
        normalized
        for urls in results
        for normalized in map(normalize_url, urls)
        if normalized is not None
    )
    logger.info("Found %d unique URLs", len(unique_urls))
    return list(unique_urls)


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
    timeout: float = 5,
    max_workers: int = 10,
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
    num_runs: int = 5,
    *,
    perplexity_model: str = DEFAULT_PERPLEXITY_MODEL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    openai_temperature: float | None = None,
    include_search_results: bool = True,
    perplexity_client: OpenAI | None = None,
    openai_client: OpenAI | None = None,
) -> list[str]:
    """Find URLs for `query`, deduplicate them and keep the online ones."""
    urls = execute_query_multiple_times(
        query,
        num_runs,
        perplexity_model=perplexity_model,
        openai_model=openai_model,
        openai_reasoning_effort=openai_reasoning_effort,
        openai_temperature=openai_temperature,
        include_search_results=include_search_results,
        perplexity_client=perplexity_client,
        openai_client=openai_client,
    )
    return check_urls_online(keep_https(urls))


class Website(BaseModel):
    name: str = Field(description="The name of the website.")
    url: str = Field(description="The url of the website.")


class WebsiteList(BaseModel):
    websites_list: list[Website] = Field(description="A short list.")


def get_top_relevant_websites(
    website_urls: list[str],
    query: str | None = None,
    *,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    openai_temperature: float | None = None,
    openai_client: OpenAI | None = None,
) -> list[Website]:
    """Ask an OpenAI model for the most relevant websites in `website_urls`.

    Pass the original `query` so the model knows what to rank for. Only
    websites from `website_urls` are returned, with the URL as given there.
    """
    if not website_urls:
        return []
    if openai_client is None:
        openai_client = _default_openai_client()

    instructions = "Select the most important websites from this JSON list."
    if query:
        instructions = (
            "Select the websites from this JSON list that are most relevant "
            f"to the following query.\nQuery: {query}"
        )
    completion = openai_client.chat.completions.parse(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{instructions}\n"
                    "Return a short list, using only URLs from the list."
                ),
            },
            {"role": "user", "content": json.dumps(website_urls)},
        ],
        response_format=WebsiteList,
        **_model_options(openai_temperature, openai_reasoning_effort),
    )
    website_list = _parsed_or_none(completion)
    if website_list is None:
        return []

    # Map the model's URLs back to the input ones, dropping any made up.
    originals: dict[str, str] = {}
    for url in website_urls:
        normalized = normalize_url(url)
        if normalized is not None:
            originals.setdefault(_dedup_key(normalized), url)
    selected: dict[str, Website] = {}
    for website in website_list.websites_list:
        normalized = normalize_url(website.url)
        original = originals.get(_dedup_key(normalized)) if normalized else None
        if original is None:
            logger.info("Dropping a URL that is not in the list: %s", website.url)
        elif original not in selected:
            selected[original] = website.model_copy(update={"url": original})
    return list(selected.values())
