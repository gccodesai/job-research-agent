"""Tests run without internet or an API key by using a fake LLM.

Run with:  pytest
"""

from types import SimpleNamespace

from agent import run_agent
from agent.prompts import build_user_message
from agent.tools import execute_tool, fetch_webpage


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def reply(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Pretends to be the OpenAI client, returning scripted replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def test_agent_runs_tool_then_returns_report(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.TOOL_FUNCTIONS",
        {"web_search": lambda query: f"results for {query}"},
    )
    client = FakeClient([
        reply(tool_calls=[tool_call("1", "web_search", '{"query": "Acme news"}')]),
        reply(content="## Role Snapshot\nGreat role."),
    ])
    seen = []
    result = run_agent("job text", "resume text", client, "fake-model", on_step=seen.append)

    assert result.report.startswith("## Role Snapshot")
    assert [s.tool for s in result.steps] == ["web_search"]
    assert seen == result.steps
    # The tool result must be sent back to the LLM on the second call.
    last_message = client.calls[1]["messages"][-1]
    assert last_message == {"role": "tool", "tool_call_id": "1", "content": "results for Acme news"}


def test_agent_forces_report_at_step_limit(monkeypatch):
    monkeypatch.setattr("agent.tools.TOOL_FUNCTIONS", {"web_search": lambda query: "ok"})
    loop = reply(tool_calls=[tool_call("x", "web_search", '{"query": "q"}')])
    client = FakeClient([loop, loop, reply(content="final report")])

    result = run_agent("job", "resume", client, "fake-model", max_steps=2)

    assert result.hit_step_limit
    assert result.report == "final report"
    assert "tools" not in client.calls[-1]  # final call offers no tools


def test_bad_tool_requests_return_errors_not_crashes():
    assert "unknown tool" in execute_tool("delete_everything", "{}")
    assert "could not parse" in execute_tool("web_search", "not json")
    assert "must start with http" in fetch_webpage("ftp://example.com")


def test_url_vs_pasted_posting():
    assert "read it first" in build_user_message("https://jobs.example.com/1", "cv")
    assert "<job_posting>" in build_user_message("Director, PMO\nAcme Corp", "cv")
