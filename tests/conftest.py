import os
import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def workspace_tmp_path():
    """Use an ordinary path because restricted Windows ACLs break pytest tmp_path."""

    root = Path(os.environ.get("NLP_RAG_TEST_TMP", ".test-artifacts")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
