import inspect
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from openai import OpenAI
from perplexity import Perplexity, PerplexityError

import ai_url_aggregator as agg

# What both OpenAI calls send with the default model options.
DEFAULT_OPENAI_KWARGS = {
    "model",
    "instructions",
    "input",
    "text_format",
    "store",
    "reasoning",
}


class FakeClient:
    """Stands in for the OpenAI and Perplexity clients and records calls.

    OpenAI is called through responses.parse and Perplexity through
    search.create. `parse` and `search` are the responses to return, or
    callables that produce them (and may raise).
    """

    def __init__(self, parse=None, search=None):
        self.calls = []
        self._responses = {"parse": parse, "search": search}
        # Expose each method only where the SDKs have it.
        self.responses = SimpleNamespace(
            parse=lambda **kwargs: self._respond("parse", kwargs)
        )
        self.search = SimpleNamespace(
            create=lambda **kwargs: self._respond("search", kwargs)
        )

    def _respond(self, method, kwargs):
        self.calls.append((method, kwargs))
        response = self._responses[method]
        return response() if callable(response) else response


def search_response(*urls):
    """A search.create() result with a title and snippet for each URL."""
    return SimpleNamespace(
        results=[
            SimpleNamespace(url=url, title=f"Title {i}", snippet=f"Snippet {i}")
            for i, url in enumerate(urls)
        ]
    )


def parse_response(parsed, refusal=None, status="completed", reason=None):
    """A responses.parse() result whose output_parsed is `parsed`."""
    content = []
    if refusal is not None:
        content.append(SimpleNamespace(type="refusal", refusal=refusal))
    return SimpleNamespace(
        output_parsed=parsed,
        # Reasoning models put a reasoning item, which has no content, first.
        output=[
            SimpleNamespace(type="reasoning", summary=[]),
            SimpleNamespace(type="message", content=content),
        ],
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
    )


def truncated_output():
    """Raise what responses.parse() raises for JSON cut off by a token limit."""
    agg.QueryList.model_validate_json('{"queries": ["a que')


def responses_reply(content, status="completed"):
    """The JSON body of a Responses API result with one message."""
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "gpt-test",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "status": "completed",
                "content": content,
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.respond(is_head=True)

    def do_GET(self):
        self.respond(is_head=False)

    def respond(self, is_head):
        port = self.server.server_address[1]
        routes = {
            "/ok": (200, None),
            "/forbidden": (403, None),
            "/head-not-allowed": (405 if is_head else 200, None),
            "/redirect": (301, "/ok"),
            "/redirect-to-missing": (302, "/missing"),
            "/redirect-elsewhere": (302, f"http://localhost:{port}/ok"),
        }
        status, location = routes.get(self.path, (404, None))
        self.send_response(status)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


class ApiHandler(BaseHTTPRequestHandler):
    """A minimal JSON API endpoint; records each request's path and body."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, json.loads(body)))
        reply = json.dumps(self.server.reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@contextmanager
def serve(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def server():
    with serve(Handler) as httpd:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def api_server():
    with serve(ApiHandler) as httpd:
        httpd.requests = []
        httpd.reply = None
        httpd.url = f"http://127.0.0.1:{httpd.server_address[1]}"
        yield httpd


def test_import_does_not_need_api_keys():
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("OPENAI_API_KEY", "PERPLEXITY_API_KEY")
    }
    subprocess.run(
        [sys.executable, "-c", "import ai_url_aggregator"], env=env, check=True
    )


def test_missing_perplexity_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    # The OpenAI key must not be used in its place.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agg._default_perplexity_client.cache_clear()
    with pytest.raises(PerplexityError, match="PERPLEXITY_API_KEY"):
        agg.search_web(["query"])


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://Example.COM/Path/", "https://example.com/Path"),
        ("www.example.com", "https://www.example.com"),
        ("example.org/a/", "https://example.org/a"),
        ("  HTTP://example.com:80/  ", "http://example.com"),
        ("https://example.com:443", "https://example.com"),
        ("https://example.com:8443/x", "https://example.com:8443/x"),
        ("https://example.com/#section", "https://example.com"),
        ("https://example.com/search?q=1", "https://example.com/search?q=1"),
        ("http://[::1]:8080/x", "http://[::1]:8080/x"),
        ("ftp://example.com", None),
        ("mailto:someone@example.com", None),
        ("javascript:alert(1)", None),
        ("https://user:pass@example.com", None),
        ("not a url", None),
        ("", None),
    ],
)
def test_normalize_url(url, expected):
    assert agg.normalize_url(url) == expected


@pytest.mark.parametrize(
    ("urls", "expected"),
    [
        (["http://a.com", "https://a.com"], ["https://a.com"]),
        (["https://a.com", "http://a.com"], ["https://a.com"]),
        (["https://a.com", "https://www.a.com/"], ["https://a.com"]),
        (["http://www.a.com", "https://a.com"], ["https://a.com"]),
        (
            ["www.example.com", "HTTPS://Upper.com", "ftp://x.com"],
            ["https://www.example.com", "https://upper.com"],
        ),
        (
            ["https://b.com", "https://a.com", "http://b.com/"],
            ["https://b.com", "https://a.com"],
        ),
    ],
)
def test_keep_https(urls, expected):
    assert agg.keep_https(urls) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://8.8.8.8/", True),
        ("http://127.0.0.1:8000/", False),
        ("http://localhost/", False),
        ("http://10.0.0.5/", False),
        ("http://192.168.1.1/", False),
        ("http://169.254.169.254/latest/meta-data/", False),
        ("http://[::1]/", False),
        ("http:///no-host", False),
    ],
)
def test_is_public_url(url, expected):
    assert agg._is_public_url(url) is expected


def test_check_urls_online_follows_redirects_and_falls_back_to_get(server):
    urls = [
        f"{server}/ok",
        f"{server}/forbidden",
        f"{server}/head-not-allowed",
        f"{server}/redirect",
        f"{server}/redirect-to-missing",
        f"{server}/missing",
    ]
    assert agg.check_urls_online(urls, allow_private_addresses=True) == [
        f"{server}/ok",
        f"{server}/forbidden",
        f"{server}/head-not-allowed",
        f"{server}/redirect",
    ]


def test_check_urls_online_blocks_private_addresses_by_default(server):
    assert agg.check_urls_online([f"{server}/ok"]) == []


def test_check_urls_online_checks_every_redirect_hop(server, monkeypatch):
    # Pretend 127.0.0.1 is public, so only the redirect to localhost is
    # blocked.
    monkeypatch.setattr(
        agg, "_is_public_url", lambda url: urlsplit(url).hostname == "127.0.0.1"
    )
    urls = [f"{server}/ok", f"{server}/redirect-elsewhere"]
    assert agg.check_urls_online(urls) == [f"{server}/ok"]


def test_check_urls_online_ignores_malformed_urls():
    assert agg.check_urls_online(["not a url", "http://"]) == []


def test_expand_query_adds_the_model_queries():
    openai = FakeClient(
        parse=parse_response(
            agg.QueryList(
                queries=["inmobiliarias Uruguay", "query", " ", "directorio", "x"]
            )
        )
    )

    queries = agg.expand_query(
        "query", 4, openai_model="gpt-test", openai_client=openai
    )

    assert queries == ["query", "inmobiliarias Uruguay", "directorio", "x"]
    [(_, kwargs)] = openai.calls
    assert kwargs.keys() == DEFAULT_OPENAI_KWARGS
    assert kwargs["model"] == "gpt-test"
    assert "Write 3 web search queries" in kwargs["instructions"]
    assert kwargs["input"] == "query"
    assert kwargs["text_format"] is agg.QueryList
    assert kwargs["store"] is False
    assert kwargs["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize(
    "parse",
    [parse_response(None, refusal="No."), truncated_output],
    ids=["refusal", "truncated"],
)
def test_expand_query_falls_back_to_the_query(parse, caplog):
    openai = FakeClient(parse=parse)
    assert agg.expand_query("query", openai_client=openai) == ["query"]
    assert "structured output" in caplog.text


def test_expand_query_with_one_query_skips_the_api():
    openai = FakeClient()
    assert agg.expand_query("query", 1, openai_client=openai) == ["query"]
    assert openai.calls == []


def test_expand_query_rejects_zero_queries():
    with pytest.raises(ValueError, match="num_queries"):
        agg.expand_query("query", 0, openai_client=FakeClient())


def test_search_web_sends_the_queries_in_batches():
    perplexity = FakeClient(search=search_response("https://a.example"))
    queries = [f"q{i}" for i in range(7)]

    results = agg.search_web(queries, perplexity_client=perplexity)

    assert [kwargs for _, kwargs in perplexity.calls] == [
        {
            "query": batch,
            "max_results": agg.MAX_RESULTS_PER_QUERY,
            "max_tokens": len(batch) * agg.MAX_RESULTS_PER_QUERY * agg._PAGE_TOKENS,
            "max_tokens_per_page": agg._PAGE_TOKENS,
        }
        for batch in (queries[:5], queries[5:])
    ]
    assert results == 2 * [
        agg.SearchResult(
            url="https://a.example", title="Title 0", snippet="Snippet 0"
        )
    ]


def test_search_web_passes_the_country_and_result_count():
    perplexity = FakeClient(search=search_response())

    agg.search_web(
        ["q"], max_results=7, country="UY", perplexity_client=perplexity
    )

    [(_, kwargs)] = perplexity.calls
    assert kwargs["max_results"] == 7
    assert kwargs["max_tokens"] == 7 * agg._PAGE_TOKENS
    assert kwargs["country"] == "UY"


def test_search_for_web_urls_end_to_end(server, monkeypatch):
    monkeypatch.setattr(agg, "_is_public_url", lambda url: True)
    openai = FakeClient(parse=parse_response(agg.QueryList(queries=["alt"])))
    perplexity = FakeClient(
        search=search_response(
            f"{server}/ok/",
            f"{server}/missing",
            f"{server}/redirect",
            # The same page as the first one, once normalized.
            f"{server}/ok",
        )
    )

    results = agg.search_for_web_urls(
        "query", 2, perplexity_client=perplexity, openai_client=openai
    )

    assert results == [
        agg.SearchResult(
            url=f"{server}/ok", title="Title 0", snippet="Snippet 0"
        ),
        agg.SearchResult(
            url=f"{server}/redirect", title="Title 2", snippet="Snippet 2"
        ),
    ]
    [(_, kwargs)] = perplexity.calls
    assert kwargs["query"] == ["query", "alt"]


def test_search_for_web_urls_passes_the_options_through():
    openai = FakeClient(parse=parse_response(agg.QueryList(queries=["alt"])))
    perplexity = FakeClient(search=search_response())

    assert agg.search_for_web_urls(
        "query",
        3,
        max_results=7,
        country="UY",
        openai_model="gpt-4o",
        openai_reasoning_effort=None,
        openai_temperature=0,
        perplexity_client=perplexity,
        openai_client=openai,
    ) == []

    [(_, parse_kwargs)] = openai.calls
    assert parse_kwargs["model"] == "gpt-4o"
    assert parse_kwargs["temperature"] == 0
    assert "reasoning" not in parse_kwargs
    assert "Write 2 web search queries" in parse_kwargs["instructions"]
    [(_, search_kwargs)] = perplexity.calls
    assert search_kwargs["max_results"] == 7
    assert search_kwargs["country"] == "UY"


def test_get_top_relevant_websites_ranks_and_classifies():
    selection = agg.WebsiteList(
        websites_list=[
            agg.Website(name="A", url="https://a.example/", kind="official_site"),
            agg.Website(
                name="Made up", url="https://invented.example", kind="other"
            ),
            agg.Website(name="B", url="http://b.example", kind="directory"),
            agg.Website(
                name="A again", url="https://www.a.example", kind="article"
            ),
        ]
    )
    openai = FakeClient(parse=parse_response(selection))
    results = [
        agg.SearchResult(
            url="https://www.a.example", title="A", snippet="Agency A"
        ),
        agg.SearchResult(url="https://b.example", title="B", snippet="x" * 600),
        # A plain URL works too.
        "https://c.example",
    ]

    websites = agg.get_top_relevant_websites(
        results,
        query="real estate agencies",
        openai_model="gpt-test",
        openai_client=openai,
    )

    assert websites == [
        agg.Website(name="A", url="https://www.a.example", kind="official_site"),
        agg.Website(name="B", url="https://b.example", kind="directory"),
    ]
    [(_, kwargs)] = openai.calls
    assert kwargs.keys() == DEFAULT_OPENAI_KWARGS
    assert kwargs["model"] == "gpt-test"
    assert "real estate agencies" in kwargs["instructions"]
    assert json.loads(kwargs["input"]) == [
        {"url": "https://www.a.example", "title": "A", "snippet": "Agency A"},
        {
            "url": "https://b.example",
            "title": "B",
            "snippet": "x" * agg._SNIPPET_CHARS,
        },
        {"url": "https://c.example", "title": "", "snippet": ""},
    ]
    assert kwargs["text_format"] is agg.WebsiteList
    assert kwargs["store"] is False
    assert kwargs["reasoning"] == {"effort": "low"}


def test_get_top_relevant_websites_without_urls_skips_the_api():
    openai = FakeClient()
    assert agg.get_top_relevant_websites([], openai_client=openai) == []
    assert openai.calls == []


@pytest.mark.parametrize(
    "parse",
    [parse_response(None, refusal="No."), truncated_output],
    ids=["refusal", "truncated"],
)
def test_get_top_relevant_websites_returns_nothing_without_output(parse):
    openai = FakeClient(parse=parse)
    assert agg.get_top_relevant_websites(
        ["https://a.example"], openai_client=openai
    ) == []


def test_get_top_relevant_websites_passes_model_options():
    openai = FakeClient(parse=parse_response(agg.WebsiteList(websites_list=[])))

    agg.get_top_relevant_websites(
        ["https://a.example"],
        openai_reasoning_effort=None,
        openai_temperature=0,
        openai_client=openai,
    )

    [(_, kwargs)] = openai.calls
    assert kwargs["temperature"] == 0
    assert "reasoning" not in kwargs


def test_api_kwargs_match_the_sdk_signatures():
    perplexity = FakeClient(search=search_response())
    agg.search_web(["q"], perplexity_client=perplexity)
    agg.search_web(["q"], country="UY", perplexity_client=perplexity)
    expansion = FakeClient(parse=parse_response(agg.QueryList(queries=[])))
    agg.expand_query("query", openai_client=expansion)
    ranking = FakeClient(parse=parse_response(agg.WebsiteList(websites_list=[])))
    agg.get_top_relevant_websites(
        ["https://a.example"],
        openai_reasoning_effort=None,
        openai_temperature=0,
        openai_client=ranking,
    )

    with (
        OpenAI(api_key="test") as openai_client,
        Perplexity(api_key="test") as perplexity_client,
    ):
        signatures = {
            "parse": inspect.signature(openai_client.responses.parse),
            "search": inspect.signature(perplexity_client.search.create),
        }
    # bind() only catches a renamed or removed kwarg if there is no **kwargs.
    for signature in signatures.values():
        assert all(
            parameter.kind is not parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    for method, kwargs in perplexity.calls + expansion.calls + ranking.calls:
        signatures[method].bind(**kwargs)


def test_search_web_with_the_real_perplexity_client(api_server):
    api_server.reply = {
        "id": "search_test",
        "results": [
            {"title": "A", "url": "https://a.example", "snippet": "Agency A"},
            {"title": "B", "url": "https://b.example", "snippet": "Agency B"},
        ],
    }

    with Perplexity(
        api_key="test", base_url=api_server.url, max_retries=0
    ) as client:
        results = agg.search_web(
            ["q1", "q2"], country="UY", perplexity_client=client
        )

    assert results == [
        agg.SearchResult(url="https://a.example", title="A", snippet="Agency A"),
        agg.SearchResult(url="https://b.example", title="B", snippet="Agency B"),
    ]
    [(path, body)] = api_server.requests
    assert path == "/search"
    assert body == {
        "query": ["q1", "q2"],
        "max_results": agg.MAX_RESULTS_PER_QUERY,
        "max_tokens": 2 * agg.MAX_RESULTS_PER_QUERY * agg._PAGE_TOKENS,
        "max_tokens_per_page": agg._PAGE_TOKENS,
        "country": "UY",
    }


@pytest.mark.parametrize(
    ("content", "status", "expected"),
    [
        (
            [
                {
                    "type": "output_text",
                    "text": '{"queries": ["alt one", "alt two"]}',
                    "annotations": [],
                }
            ],
            "completed",
            ["query", "alt one", "alt two"],
        ),
        ([{"type": "refusal", "refusal": "No."}], "completed", ["query"]),
        (
            [
                {
                    "type": "output_text",
                    "text": '{"queries": ["alt',
                    "annotations": [],
                }
            ],
            "incomplete",
            ["query"],
        ),
    ],
    ids=["valid", "refusal", "truncated"],
)
def test_expand_query_with_the_real_openai_client(
    api_server, content, status, expected
):
    api_server.reply = responses_reply(content, status)

    with OpenAI(api_key="test", base_url=api_server.url, max_retries=0) as client:
        queries = agg.expand_query("query", openai_client=client)

    assert queries == expected
    [(path, body)] = api_server.requests
    assert path == "/responses"
    assert body["input"] == "query"
    assert body["store"] is False
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["type"] == "json_schema"
