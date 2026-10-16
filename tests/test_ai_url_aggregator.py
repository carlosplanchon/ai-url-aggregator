import itertools
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from openai import OpenAIError

import ai_url_aggregator as agg


class FakeClient:
    """Stands in for openai.OpenAI and records chat.completions calls.

    `create` and `parse` are the responses to return, or callables that
    produce them (and may raise).
    """

    def __init__(self, create=None, parse=None):
        self.calls = []
        self._responses = {"create": create, "parse": parse}
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        return self._respond("create", kwargs)

    def parse(self, **kwargs):
        return self._respond("parse", kwargs)

    def _respond(self, method, kwargs):
        self.calls.append((method, kwargs))
        response = self._responses[method]
        return response() if callable(response) else response


def perplexity_response(content, search_results=()):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        search_results=[{"title": "", "url": url} for url in search_results],
    )


def parse_response(parsed, refusal=None):
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


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


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_import_does_not_need_api_keys():
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("OPENAI_API_KEY", "PERPLEXITY_API_KEY")
    }
    subprocess.run(
        [sys.executable, "-c", "import ai_url_aggregator"], env=env, check=True
    )


def test_perplexity_never_falls_back_to_the_openai_key(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agg._default_perplexity_client.cache_clear()
    with pytest.raises(RuntimeError, match="PERPLEXITY_API_KEY"):
        agg.query_models("query", openai_client=FakeClient())


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
    ("text", "expected"),
    [
        (
            "<think>try https://guess.example</think>\nSee https://a.example",
            "See https://a.example",
        ),
        ("reasoning without an opening tag</think>Answer", "Answer"),
        ("<think>cut off mid-thought", ""),
        ("Plain answer", "Plain answer"),
    ],
)
def test_strip_reasoning(text, expected):
    assert agg._strip_reasoning(text) == expected


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


def test_query_models_extracts_urls_from_the_answer_and_adds_sources():
    perplexity = FakeClient(
        create=perplexity_response(
            "<think>maybe https://guess.example</think>See https://a.example",
            search_results=["https://source.example"],
        )
    )
    openai = FakeClient(
        parse=parse_response(agg.UrlList(urls=["https://a.example"]))
    )

    urls = agg.query_models(
        "query",
        perplexity_model="sonar-test",
        openai_model="gpt-test",
        perplexity_client=perplexity,
        openai_client=openai,
    )

    assert urls == ["https://a.example", "https://source.example"]
    [(_, create_kwargs)] = perplexity.calls
    assert create_kwargs["model"] == "sonar-test"
    assert create_kwargs["messages"] == [{"role": "user", "content": "query"}]
    [(_, parse_kwargs)] = openai.calls
    assert parse_kwargs["model"] == "gpt-test"
    assert parse_kwargs["response_format"] is agg.UrlList
    assert parse_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in parse_kwargs
    system, user = parse_kwargs["messages"]
    assert system["role"] == "system"
    assert user == {"role": "user", "content": "See https://a.example"}


def test_query_models_survives_a_refusal(caplog):
    perplexity = FakeClient(
        create=perplexity_response(
            "Answer", search_results=["https://source.example"]
        )
    )
    openai = FakeClient(parse=parse_response(None, refusal="I can't help."))

    assert agg.query_models(
        "query", perplexity_client=perplexity, openai_client=openai
    ) == ["https://source.example"]
    assert "I can't help." in caplog.text
    assert agg.query_models(
        "query",
        include_search_results=False,
        perplexity_client=perplexity,
        openai_client=openai,
    ) == []


def test_query_models_skips_extraction_without_an_answer():
    perplexity = FakeClient(create=perplexity_response("<think>no answer"))
    openai = FakeClient()

    assert agg.query_models(
        "query", perplexity_client=perplexity, openai_client=openai
    ) == []
    assert openai.calls == []


def test_query_models_passes_custom_model_options():
    openai = FakeClient(parse=parse_response(agg.UrlList(urls=[])))

    agg.query_models(
        "query",
        openai_reasoning_effort="none",
        openai_temperature=0.7,
        perplexity_client=FakeClient(create=perplexity_response("Answer")),
        openai_client=openai,
    )

    [(_, kwargs)] = openai.calls
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["temperature"] == 0.7


def test_search_for_web_urls_passes_model_options_through():
    openai = FakeClient(parse=parse_response(agg.UrlList(urls=[])))

    agg.search_for_web_urls(
        "query",
        num_runs=2,
        openai_model="gpt-4o",
        openai_reasoning_effort=None,
        openai_temperature=0,
        perplexity_client=FakeClient(create=perplexity_response("Answer")),
        openai_client=openai,
    )

    assert len(openai.calls) == 2
    for _, kwargs in openai.calls:
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0
        assert "reasoning_effort" not in kwargs


def test_execute_query_multiple_times_merges_and_normalizes_runs():
    perplexity = FakeClient(
        create=perplexity_response(
            "Answer", search_results=["https://b.example"]
        )
    )
    openai = FakeClient(
        parse=parse_response(
            agg.UrlList(
                urls=["https://a.example/", "HTTPS://A.example", "not a url"]
            )
        )
    )

    urls = agg.execute_query_multiple_times(
        "query", num_runs=3, perplexity_client=perplexity, openai_client=openai
    )

    assert urls == ["https://a.example", "https://b.example"]
    assert len(perplexity.calls) == 3


def test_execute_query_multiple_times_skips_failed_runs(caplog):
    attempts = itertools.count()

    def create():
        if next(attempts) == 0:
            raise OpenAIError("temporary failure")
        return perplexity_response("", search_results=["https://a.example"])

    urls = agg.execute_query_multiple_times(
        "query",
        num_runs=3,
        perplexity_client=FakeClient(create=create),
        openai_client=FakeClient(),
    )

    assert urls == ["https://a.example"]
    assert "temporary failure" in caplog.text


def test_execute_query_multiple_times_does_not_hide_bugs():
    attempts = itertools.count()

    def create():
        if next(attempts) == 0:
            raise TypeError("bug")
        return perplexity_response("", search_results=["https://a.example"])

    with pytest.raises(TypeError, match="bug"):
        agg.execute_query_multiple_times(
            "query",
            num_runs=3,
            perplexity_client=FakeClient(create=create),
            openai_client=FakeClient(),
        )


def test_execute_query_multiple_times_raises_when_every_run_fails():
    def create():
        raise OpenAIError("model not found")

    with pytest.raises(OpenAIError, match="model not found"):
        agg.execute_query_multiple_times(
            "query",
            num_runs=2,
            perplexity_client=FakeClient(create=create),
            openai_client=FakeClient(),
        )


def test_execute_query_multiple_times_rejects_zero_runs():
    with pytest.raises(ValueError, match="num_runs"):
        agg.execute_query_multiple_times(
            "query",
            num_runs=0,
            perplexity_client=FakeClient(),
            openai_client=FakeClient(),
        )


def test_get_top_relevant_websites_ranks_for_the_query():
    selection = agg.WebsiteList(
        websites_list=[
            agg.Website(name="A", url="https://a.example/"),
            agg.Website(name="Made up", url="https://invented.example"),
            agg.Website(name="B", url="http://b.example"),
            agg.Website(name="A again", url="https://www.a.example"),
        ]
    )
    openai = FakeClient(parse=parse_response(selection))
    input_urls = [
        "https://www.a.example",
        "https://b.example",
        "https://c.example",
    ]

    websites = agg.get_top_relevant_websites(
        input_urls,
        query="real estate agencies",
        openai_model="gpt-test",
        openai_client=openai,
    )

    assert websites == [
        agg.Website(name="A", url="https://www.a.example"),
        agg.Website(name="B", url="https://b.example"),
    ]
    [(_, kwargs)] = openai.calls
    assert kwargs["model"] == "gpt-test"
    assert kwargs["reasoning_effort"] == "low"
    assert "temperature" not in kwargs
    system, user = kwargs["messages"]
    assert system["role"] == "system"
    assert "real estate agencies" in system["content"]
    assert json.loads(user["content"]) == input_urls


def test_get_top_relevant_websites_without_urls_skips_the_api():
    openai = FakeClient()
    assert agg.get_top_relevant_websites([], openai_client=openai) == []
    assert openai.calls == []


def test_get_top_relevant_websites_returns_nothing_on_refusal():
    openai = FakeClient(parse=parse_response(None, refusal="No."))
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
    assert "reasoning_effort" not in kwargs


def test_search_for_web_urls_end_to_end(server, monkeypatch):
    monkeypatch.setattr(agg, "_is_public_url", lambda url: True)
    perplexity = FakeClient(
        create=perplexity_response(
            "Answer", search_results=[f"{server}/redirect"]
        )
    )
    openai = FakeClient(
        parse=parse_response(
            agg.UrlList(urls=[f"{server}/ok/", f"{server}/missing"])
        )
    )

    urls = agg.search_for_web_urls(
        "query", num_runs=2, perplexity_client=perplexity, openai_client=openai
    )

    assert urls == [f"{server}/ok", f"{server}/redirect"]
    assert len(perplexity.calls) == 2
