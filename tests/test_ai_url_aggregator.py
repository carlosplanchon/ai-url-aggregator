import inspect
import itertools
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
from openai import OpenAI, OpenAIError

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
    """Stands in for openai.OpenAI and records the API calls it gets.

    Perplexity is called through responses.create and OpenAI through
    responses.parse. `create` and `parse` are the responses to return, or
    callables that produce them (and may raise).
    """

    def __init__(self, create=None, parse=None):
        self.calls = []
        self._responses = {"create": create, "parse": parse}
        # No chat attribute: neither API is called through Chat Completions.
        self.responses = SimpleNamespace(create=self.create, parse=self.parse)

    def create(self, **kwargs):
        return self._respond("create", kwargs)

    def parse(self, **kwargs):
        return self._respond("parse", kwargs)

    def _respond(self, method, kwargs):
        self.calls.append((method, kwargs))
        response = self._responses[method]
        return response() if callable(response) else response


def perplexity_response(text, search_results=(), status="completed"):
    """A Perplexity Agent API result: its search results, then its answer.

    The results are dicts, as the OpenAI SDK leaves them.
    """
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="search_results",
                results=[{"title": "", "url": url} for url in search_results],
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            ),
        ],
        output_text=text,
        status=status,
        error=None,
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
    agg.UrlList.model_validate_json('{"urls": ["https://a.ex')


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


class ResponsesHandler(BaseHTTPRequestHandler):
    """A minimal Responses API endpoint; records each request's path and body."""

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
def responses_server():
    with serve(ResponsesHandler) as httpd:
        httpd.requests = []
        httpd.reply = None
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


def test_search_result_urls_reads_dicts_and_objects():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", summary=[]),
            # The OpenAI SDK leaves the entries of these items as dicts.
            SimpleNamespace(
                type="search_results",
                results=[{"url": "https://a.example"}, {"title": "No URL"}],
            ),
            SimpleNamespace(
                type="fetch_url_results",
                contents=[{"url": "https://b.example"}],
            ),
            # Perplexity's SDK makes objects.
            SimpleNamespace(
                type="search_results",
                results=[
                    SimpleNamespace(url="https://c.example"),
                    SimpleNamespace(url=None),
                ],
            ),
            SimpleNamespace(type="message", content=[]),
        ]
    )

    assert agg._search_result_urls(response) == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]


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
            "See https://a.example[web:1]\n",
            search_results=["https://source.example"],
        )
    )
    openai = FakeClient(
        parse=parse_response(agg.UrlList(urls=["https://a.example"]))
    )

    urls = agg.query_models(
        "query",
        openai_model="gpt-test",
        perplexity_client=perplexity,
        openai_client=openai,
    )

    assert urls == ["https://a.example", "https://source.example"]
    assert perplexity.calls == [
        (
            "create",
            {
                "input": f"query\n\n{agg._URL_REQUEST}",
                "extra_body": {"preset": "medium"},
            },
        )
    ]
    [(_, parse_kwargs)] = openai.calls
    assert parse_kwargs.keys() == DEFAULT_OPENAI_KWARGS
    assert parse_kwargs["model"] == "gpt-test"
    assert parse_kwargs["input"] == "See https://a.example"
    assert parse_kwargs["text_format"] is agg.UrlList
    assert parse_kwargs["store"] is False
    assert parse_kwargs["reasoning"] == {"effort": "low"}


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


def test_query_models_survives_truncated_output(caplog):
    perplexity = FakeClient(
        create=perplexity_response(
            "Answer", search_results=["https://source.example"]
        )
    )

    urls = agg.query_models(
        "query",
        perplexity_client=perplexity,
        openai_client=FakeClient(parse=truncated_output),
    )

    assert urls == ["https://source.example"]
    assert "invalid structured output" in caplog.text


def test_query_models_logs_why_there_is_no_output(caplog):
    openai = FakeClient(
        parse=parse_response(
            None, status="incomplete", reason="max_output_tokens"
        )
    )

    assert agg.query_models(
        "query",
        include_search_results=False,
        perplexity_client=FakeClient(create=perplexity_response("Answer")),
        openai_client=openai,
    ) == []
    assert "incomplete" in caplog.text
    assert "max_output_tokens" in caplog.text


def test_query_models_skips_extraction_without_an_answer():
    perplexity = FakeClient(create=perplexity_response(" \n"))
    openai = FakeClient()

    assert agg.query_models(
        "query", perplexity_client=perplexity, openai_client=openai
    ) == []
    assert openai.calls == []


def test_query_models_warns_about_an_unfinished_perplexity_answer(caplog):
    perplexity = FakeClient(
        create=perplexity_response(
            "", search_results=["https://a.example"], status="incomplete"
        )
    )

    assert agg.query_models(
        "query", perplexity_client=perplexity, openai_client=FakeClient()
    ) == ["https://a.example"]
    assert "incomplete" in caplog.text


def test_query_models_raises_when_perplexity_fails():
    response = perplexity_response("", status="failed")
    response.error = SimpleNamespace(message="Agent crashed")

    with pytest.raises(OpenAIError, match="Agent crashed"):
        agg.query_models(
            "query",
            perplexity_client=FakeClient(create=response),
            openai_client=FakeClient(),
        )


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
    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["temperature"] == 0.7


def test_search_for_web_urls_passes_model_options_through():
    perplexity = FakeClient(create=perplexity_response("Answer"))
    openai = FakeClient(parse=parse_response(agg.UrlList(urls=[])))

    agg.search_for_web_urls(
        "query",
        num_runs=2,
        perplexity_preset="low",
        perplexity_model="openai/gpt-test",
        openai_model="gpt-4o",
        openai_reasoning_effort=None,
        openai_temperature=0,
        perplexity_client=perplexity,
        openai_client=openai,
    )

    assert len(perplexity.calls) == 2
    for _, kwargs in perplexity.calls:
        assert kwargs["model"] == "openai/gpt-test"
        assert kwargs["extra_body"] == {"preset": "low"}
    assert len(openai.calls) == 2
    for _, kwargs in openai.calls:
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0
        assert "reasoning" not in kwargs


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
    assert kwargs.keys() == DEFAULT_OPENAI_KWARGS
    assert kwargs["model"] == "gpt-test"
    assert "real estate agencies" in kwargs["instructions"]
    assert json.loads(kwargs["input"]) == input_urls
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
    perplexity = FakeClient(create=perplexity_response("Answer"))
    extraction = FakeClient(parse=parse_response(agg.UrlList(urls=[])))
    # The preset's model, then another one.
    for model in (None, "openai/gpt-test"):
        agg.query_models(
            "query",
            perplexity_model=model,
            perplexity_client=perplexity,
            openai_client=extraction,
        )
    ranking = FakeClient(parse=parse_response(agg.WebsiteList(websites_list=[])))
    agg.get_top_relevant_websites(
        ["https://a.example"],
        openai_reasoning_effort=None,
        openai_temperature=0,
        openai_client=ranking,
    )

    with OpenAI(api_key="test") as client:
        signatures = {
            "create": inspect.signature(client.responses.create),
            "parse": inspect.signature(client.responses.parse),
        }
    # bind() only catches a renamed or removed kwarg if there is no **kwargs.
    for signature in signatures.values():
        assert all(
            parameter.kind is not parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    for method, kwargs in perplexity.calls + extraction.calls + ranking.calls:
        signatures[method].bind(**kwargs)


@pytest.mark.parametrize(
    ("content", "status", "expected"),
    [
        (
            [
                {
                    "type": "output_text",
                    "text": '{"urls": ["https://a.example"]}',
                    "annotations": [],
                }
            ],
            "completed",
            ["https://a.example"],
        ),
        ([{"type": "refusal", "refusal": "No."}], "completed", []),
        (
            [
                {
                    "type": "output_text",
                    "text": '{"urls": ["https://a.ex',
                    "annotations": [],
                }
            ],
            "incomplete",
            [],
        ),
    ],
    ids=["valid", "refusal", "truncated"],
)
def test_query_models_with_the_real_openai_client(
    responses_server, content, status, expected
):
    responses_server.reply = responses_reply(content, status)
    url = f"http://127.0.0.1:{responses_server.server_address[1]}"

    with OpenAI(api_key="test", base_url=url, max_retries=0) as client:
        urls = agg.query_models(
            "query",
            include_search_results=False,
            perplexity_client=FakeClient(create=perplexity_response("Answer")),
            openai_client=client,
        )

    assert urls == expected
    [(path, body)] = responses_server.requests
    assert path == "/responses"
    assert body["input"] == "Answer"
    assert body["store"] is False
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["type"] == "json_schema"


def test_query_models_with_the_real_perplexity_client(responses_server):
    reply = responses_reply(
        [
            {
                "type": "output_text",
                "text": "See https://a.example[web:1]",
                "annotations": [],
            }
        ]
    )
    # Reasoning and sources come before the answer; the SDK doesn't know the
    # search_results and fetch_url_results items.
    reply["output"][:0] = [
        {
            "type": "reasoning",
            "id": "rs_test",
            "summary": [
                {"type": "summary_text", "text": "Try https://guess.example"}
            ],
        },
        {
            "type": "search_results",
            "results": [
                {"id": 1, "title": "Source", "url": "https://source.example"}
            ],
        },
        {
            "type": "fetch_url_results",
            "contents": [{"title": "", "url": "https://fetched.example"}],
        },
    ]
    responses_server.reply = reply
    port = responses_server.server_address[1]
    # The local server, under the path of Perplexity's API.
    base_url = f"http://127.0.0.1:{port}{urlsplit(agg.PERPLEXITY_BASE_URL).path}"
    openai = FakeClient(
        parse=parse_response(agg.UrlList(urls=["https://a.example"]))
    )

    with OpenAI(api_key="test", base_url=base_url, max_retries=0) as client:
        urls = agg.query_models(
            "query", perplexity_client=client, openai_client=openai
        )

    assert urls == [
        "https://a.example",
        "https://source.example",
        "https://fetched.example",
    ]
    [(path, body)] = responses_server.requests
    assert path == "/v1/responses"
    assert body == {
        "input": f"query\n\n{agg._URL_REQUEST}",
        "preset": agg.DEFAULT_PERPLEXITY_PRESET,
    }
    [(_, parse_kwargs)] = openai.calls
    assert parse_kwargs["input"] == "See https://a.example"


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
