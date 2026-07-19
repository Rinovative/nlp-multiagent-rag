from io import StringIO

import pytest

from src import quota
from src.cli import cli_quota


class RecordingBackend(quota.memory.InMemoryQuotaBackend):
    def __init__(self):
        super().__init__()
        self.events = []

    def inspect(self, *, now=None):
        self.events.append("inspect")
        return super().inspect(now=now)

    def set_limits(self, limits):
        self.events.append("set_limits")
        super().set_limits(limits)

    def set_enabled(self, enabled):
        self.events.append("enable" if enabled else "disable")
        super().set_enabled(enabled)


def test_cli_inspects_before_mutating_all_limits():
    backend = RecordingBackend()
    stdout = StringIO()
    stderr = StringIO()

    result = cli_quota.run(
        [
            "--redis-url",
            "redis://private",
            "set-limits",
            "--daily-requests",
            "10",
            "--monthly-requests",
            "100",
            "--daily-tokens",
            "1000",
            "--monthly-tokens",
            "10000",
            "--session-requests",
            "3",
            "--session-window-seconds",
            "3600",
        ],
        stdout=stdout,
        stderr=stderr,
        backend_factory=lambda *_args, **_kwargs: backend,
    )

    assert result == 0
    assert backend.events == ["inspect", "set_limits", "inspect"]
    assert '"daily_requests": 10' in stdout.getvalue()
    assert "redis://private" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_cli_requires_private_redis_configuration():
    stderr = StringIO()
    result = cli_quota.run(["inspect"], environ={}, stdout=StringIO(), stderr=stderr)
    assert result == 2
    assert "REDIS_URL" in stderr.getvalue()


def test_cli_reads_redis_url_from_local_dotenv(workspace_tmp_path, monkeypatch):
    (workspace_tmp_path / ".env").write_text(
        "REDIS_URL=redis://dotenv-private\n", encoding="utf-8"
    )
    monkeypatch.chdir(workspace_tmp_path)
    monkeypatch.delenv("REDIS_URL", raising=False)
    backend = RecordingBackend()
    seen_urls = []

    result = cli_quota.run(
        ["inspect"],
        stdout=StringIO(),
        stderr=StringIO(),
        backend_factory=lambda redis_url, **_kwargs: (
            seen_urls.append(redis_url) or backend
        ),
    )

    assert result == 0
    assert seen_urls == ["redis://dotenv-private"]


@pytest.mark.parametrize(("command", "enabled"), [("disable", False), ("enable", True)])
def test_cli_changes_immediate_availability(command, enabled):
    backend = RecordingBackend()
    backend.set_limits(
        quota.contracts.QuotaLimits(
            daily_requests=10,
            monthly_requests=100,
            daily_tokens=1000,
            monthly_tokens=10_000,
            session_requests=3,
            session_window_seconds=3600,
        )
    )
    stdout = StringIO()

    result = cli_quota.run(
        [command],
        environ={"REDIS_URL": "redis://private"},
        stdout=stdout,
        backend_factory=lambda *_args, **_kwargs: backend,
    )

    assert result == 0
    assert backend.inspect().enabled is enabled
    assert command in backend.events
    assert "Before:" in stdout.getvalue()
    assert "After:" in stdout.getvalue()
    assert "redis://private" not in stdout.getvalue()
