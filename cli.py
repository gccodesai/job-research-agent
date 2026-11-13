"""Run the agent from the terminal (handy while learning and debugging).

Examples:
    python cli.py --job sample_data/sample_job.txt --resume sample_data/sample_resume.txt
    python cli.py --job https://careers.example.com/jobs/123 --resume my_resume.txt --provider groq
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent import PROVIDERS, make_client, model_for, run_agent


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Job Posting Research Agent")
    parser.add_argument("--job", required=True, help="Job posting URL or path to a text file")
    parser.add_argument("--resume", required=True, help="Path to resume text file")
    parser.add_argument("--provider", default="gemini", choices=list(PROVIDERS))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", default="report.md", help="Where to save the report")
    args = parser.parse_args()

    job = args.job if args.job.startswith("http") else Path(args.job).read_text()
    resume = Path(args.resume).read_text()

    client = make_client(args.provider)
    result = run_agent(
        job, resume, client, model_for(args.provider),
        max_steps=args.max_steps,
        on_step=lambda s: print(f"  -> {s.tool}({s.arguments})"),
    )
    Path(args.out).write_text(result.report)
    print(f"\n{result.report}\n\nSaved to {args.out}")


if __name__ == "__main__":
    main()
