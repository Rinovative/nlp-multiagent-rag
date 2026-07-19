"""
===============================================================================
cli_quota.py
===============================================================================
Inspect and update owner-controlled OpenAI quota settings.

Responsibilities:
  - Parse owner-only quota commands and resolve private Redis configuration.
  - Print safe before-and-after quota snapshots for mutating commands.

Design principles:
  - Keep administration explicit, import-safe, and scriptable.
  - Inject output streams and backend construction for deterministic tests.

Boundaries:
  - Quota state transitions remain in the quota package.
  - Never prints Redis credentials or connects during package import.

Notes:
  - Execute with ``python -m src.cli.cli_quota``.
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import TextIO

from dotenv import load_dotenv

from src import quota

__all__: list[str] = []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or update the private Redis-backed OpenAI allowance."
    )
    parser.add_argument(
        "--redis-url",
        help="Redis URL; defaults to the REDIS_URL environment variable.",
    )
    parser.add_argument(
        "--key-prefix",
        default="nlp-rag:{openai-quota}",
        help="Redis key namespace shared with the application.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Show active UTC periods, limits, and usage.")
    set_limits = commands.add_parser(
        "set-limits", help="Replace all hard limits without changing enabled state."
    )
    set_limits.add_argument("--daily-requests", type=int, required=True)
    set_limits.add_argument("--monthly-requests", type=int, required=True)
    set_limits.add_argument("--daily-tokens", type=int, required=True)
    set_limits.add_argument("--monthly-tokens", type=int, required=True)
    set_limits.add_argument("--session-requests", type=int, required=True)
    set_limits.add_argument("--session-window-seconds", type=int, required=True)
    commands.add_parser("disable", help="Disable OpenAI authorization immediately.")
    commands.add_parser("enable", help="Enable configured OpenAI authorization.")
    return parser


def _snapshot_json(snapshot: quota.contracts.QuotaUsageSnapshot) -> str:
    return json.dumps(asdict(snapshot), indent=2, sort_keys=True)


def run(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    backend_factory: Callable[..., quota.contracts.QuotaBackend] = (
        quota.redis.RedisQuotaBackend
    ),
) -> int:
    """Run one owner quota-administration command with injectable boundaries.

    Parameters
    ----------
    argv
        Command arguments excluding the executable name; defaults to ``sys.argv``.
    environ
        Optional environment mapping used after loading a local ``.env`` file.
    stdout
        Stream for safe inspection and before-and-after snapshots.
    stderr
        Stream for validation and quota-domain errors.
    backend_factory
        Factory receiving the private Redis URL and configured key prefix.

    Returns
    -------
    int
        ``0`` on success, ``1`` for quota or value errors, or ``2`` when the
        private Redis URL is missing.

    Notes
    -----
    Mutating commands inspect the backend before and after the requested change.
    Redis credentials are never included in output.
    """

    args = _parser().parse_args(argv)
    if environ is None:
        load_dotenv(dotenv_path=Path.cwd() / ".env")
    environment = os.environ if environ is None else environ
    redis_url = args.redis_url or environment.get("REDIS_URL")
    if not redis_url:
        stderr.write("REDIS_URL or --redis-url is required.\n")
        return 2
    backend = backend_factory(redis_url, key_prefix=args.key_prefix)
    try:
        before = backend.inspect()
        stdout.write("Before:\n")
        stdout.write(f"{_snapshot_json(before)}\n")
        if args.command == "inspect":
            return 0
        if args.command == "set-limits":
            backend.set_limits(
                quota.contracts.QuotaLimits(
                    daily_requests=args.daily_requests,
                    monthly_requests=args.monthly_requests,
                    daily_tokens=args.daily_tokens,
                    monthly_tokens=args.monthly_tokens,
                    session_requests=args.session_requests,
                    session_window_seconds=args.session_window_seconds,
                )
            )
        elif args.command == "disable":
            backend.set_enabled(False)
        elif args.command == "enable":
            backend.set_enabled(True)
        after = backend.inspect()
    except (ValueError, quota.contracts.QuotaError) as exc:
        stderr.write(f"Quota command failed: {exc}\n")
        return 1
    stdout.write("After:\n")
    stdout.write(f"{_snapshot_json(after)}\n")
    return 0


def main() -> None:
    """Execute the quota command and terminate with its returned status code.

    Notes
    -----
    This translates :func:`run`'s status to ``SystemExit``. Argument parsing may
    also exit for invalid command syntax. Importing :mod:`src.cli` does not import
    this executable module.
    """

    raise SystemExit(run())


if __name__ == "__main__":
    main()
