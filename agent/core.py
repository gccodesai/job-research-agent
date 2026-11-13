"""The agent loop: the heart of the project.

    1. Send the conversation + tool descriptions to the LLM.
    2. If the LLM asks for tools, run them, add the results, and go to 1.
    3. If the LLM answers with text, that's the final report. Done.

A step limit stops the agent from looping forever (and from burning through
free-tier rate limits).
"""

from dataclasses import dataclass, field
from typing import Callable

from .prompts import FORCE_FINISH_PROMPT, SYSTEM_PROMPT, build_user_message
from .tools import TOOL_SCHEMAS, execute_tool


@dataclass
class Step:
    """One tool call the agent made (shown live in the UI)."""

    tool: str
    arguments: str
    result_preview: str


@dataclass
class AgentResult:
    report: str
    steps: list[Step] = field(default_factory=list)
    hit_step_limit: bool = False


def run_agent(
    job_posting: str,
    resume: str,
    client,
    model: str,
    max_steps: int = 8,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    """Research a job posting against a resume and return a prep report."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(job_posting, resume)},
    ]
    steps: list[Step] = []

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message

        # No tool calls means the LLM has written its final answer.
        if not message.tool_calls:
            return AgentResult(report=message.content or "", steps=steps)

        # Record the LLM's request so it remembers what it asked for.
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        # Run each requested tool and send the result back.
        for tc in message.tool_calls:
            result = execute_tool(tc.function.name, tc.function.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
            step = Step(tc.function.name, tc.function.arguments, result[:300])
            steps.append(step)
            if on_step:
                on_step(step)

    # Out of steps: ask for the report with no tools available.
    messages.append({"role": "user", "content": FORCE_FINISH_PROMPT})
    response = client.chat.completions.create(model=model, messages=messages)
    return AgentResult(
        report=response.choices[0].message.content or "",
        steps=steps,
        hit_step_limit=True,
    )
