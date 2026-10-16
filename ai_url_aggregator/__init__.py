"""Search the web for URLs with Perplexity and filter them with OpenAI."""

import ipaddress
import json
import logging
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import cache, partial
from typing import TypeVar
from urllib.parse import urlsplit, urlunsplit

import requests
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

PERPLEXITY_BASE_URL = "https://api.perplexity.ai/v1"
# The Agent API's counterpart of the sonar-reasoning-pro model.
DEFAULT_PERPLEXITY_PRESET = "medium"
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


# The Agent API presets cite sources by number ([web:1]) instead of by URL,
# so the answer would otherwise have few URLs to extract.
_URL_REQUEST = "Include the full website URL of each item you mention."
_CITATION_RE = re.compile(r"\[[a-z_]+:\d+\]")

# Perplexity's output items that list web pages, and the field with the list.
_SOURCE_FIELDS = {"search_results": "results", "fetch_url_results": "contents"}


def _search_result_urls(response) -> list[str]:
    """URLs of the pages Perplexity found or fetched while answering."""
    urls = []
    for item in response.output:
        field = _SOURCE_FIELDS.get(item.type)
        if field is None:
            continue
        # The OpenAI SDK doesn't know these items, so it leaves their entries
        # as dicts; Perplexity's SDK makes objects.
        for source in getattr(item, field, None) or []:
            if isinstance(source, dict):
                url = source.get("url")
            else:
                url = getattr(source, "url", None)
            if isinstance(url, str):
                urls.append(url)
    return urls


def _ask_perplexity(
    client: OpenAI, query: str, *, preset: str, model: str | None
):
    """Ask Perplexity's Agent API about `query`, requesting URLs too.

    `model`, if given, replaces the preset's. A failed answer raises
    OpenAIError; an otherwise unfinished one is logged and returned as is.
    """
    options = {} if model is None else {"model": model}
    response = client.responses.create(
        input=f"{query}\n\n{_URL_REQUEST}",
        # The OpenAI SDK has no parameter for Perplexity's presets.
        extra_body={"preset": preset},
        **options,
    )
    if response.status == "failed":
        message = getattr(response.error, "message", None)
        raise OpenAIError(f"Perplexity failed to answer: {message}")
    if response.status != "completed":
        logger.warning(
            "Perplexity did not complete its answer (status: %s)",
            response.status,
        )
    return response


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


def query_models(
    query: str,
    *,
    perplexity_preset: str = DEFAULT_PERPLEXITY_PRESET,
    perplexity_model: str | None = None,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    openai_temperature: float | None = None,
    include_search_results: bool = True,
    perplexity_client: OpenAI | None = None,
    openai_client: OpenAI | None = None,
) -> list[str]:
    """Ask Perplexity about `query` and return the URLs it comes up with.

    Perplexity answers through its Agent API with `perplexity_preset`;
    `perplexity_model`, if given, replaces the preset's model. An OpenAI
    model extracts the URLs mentioned in the answer; if it refuses or its
    output is cut off, that is logged and no URLs are extracted. With
    `include_search_results`, the pages Perplexity found are added too.
    For an OpenAI model without reasoning, such as gpt-4o, pass
    `openai_reasoning_effort=None` (and optionally `openai_temperature=0`).
    """
    if perplexity_client is None:
        perplexity_client = _default_perplexity_client()
    if openai_client is None:
        openai_client = _default_openai_client()

    response = _ask_perplexity(
        perplexity_client,
        query,
        preset=perplexity_preset,
        model=perplexity_model,
    )
    answer = _CITATION_RE.sub("", response.output_text).strip()
    logger.debug("Perplexity answer:\n%s", answer)

    urls: list[str] = []
    if answer:
        url_list = _parse_structured(
            openai_client,
            model=openai_model,
            instructions=(
                "Extract every website URL mentioned in the text. "
                "Copy them exactly and do not invent any."
            ),
            user_input=answer,
            text_format=UrlList,
            temperature=openai_temperature,
            reasoning_effort=openai_reasoning_effort,
        )
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
    perplexity_preset: str = DEFAULT_PERPLEXITY_PRESET,
    perplexity_model: str | None = None,
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
        perplexity_preset=perplexity_preset,
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
    perplexity_preset: str = DEFAULT_PERPLEXITY_PRESET,
    perplexity_model: str | None = None,
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
        perplexity_preset=perplexity_preset,
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
    If the model refuses or its output is cut off, that is logged and the
    result is empty.
    """
    if not website_urls:
        return []
    if openai_client is None:
        openai_client = _default_openai_client()

    task = "Select the most important websites from this JSON list."
    if query:
        task = (
            "Select the websites from this JSON list that are most relevant "
            f"to the following query.\nQuery: {query}"
        )
    website_list = _parse_structured(
        openai_client,
        model=openai_model,
        instructions=(
            f"{task}\nReturn a short list, using only URLs from the list."
        ),
        user_input=json.dumps(website_urls),
        text_format=WebsiteList,
        temperature=openai_temperature,
        reasoning_effort=openai_reasoning_effort,
    )
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
