import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import vectorstore

faiss_store_module = vectorstore.faiss
CorruptSnapshotError = vectorstore.faiss.CorruptSnapshotError
DimensionMismatchError = vectorstore.faiss.DimensionMismatchError
DuplicateChunkIDError = vectorstore.faiss.DuplicateChunkIDError
FAISSStore = vectorstore.faiss.FAISSStore
FAISSStoreError = vectorstore.faiss.FAISSStoreError
IncompatibleSnapshotError = vectorstore.faiss.IncompatibleSnapshotError
InvalidVectorRecordError = vectorstore.faiss.InvalidVectorRecordError


DIMENSION = 3
MODEL = "test-embedding-model"


def embedded_chunk(chunk_id, vector, *, page=1, labels=None):
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "metadata": {
            "schema_version": 1,
            "length_unit": "characters",
            "document_id": "doc-a",
            "document_title": "Test document",
            "source_type": "paragraph",
            "source_sequence": page - 1,
            "chunk_sequence": page - 1,
            "part_index": 0,
            "part_count": 1,
            "page_number": page,
            "labels": labels or ["one", "two"],
            "nested": {"active": True, "score": 1.5},
        },
        "embedding": vector,
    }


def generation_directory(snapshot_directory: Path) -> Path:
    generation = (snapshot_directory / "CURRENT").read_text(encoding="utf-8")
    return snapshot_directory / "snapshots" / generation


def manifest(snapshot_directory: Path) -> tuple[Path, dict]:
    path = generation_directory(snapshot_directory) / "manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_snapshot_reloads_records_and_metadata(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks(
        [
            embedded_chunk("chunk-a", [0.0, 0.0, 0.0], page=1),
            embedded_chunk("chunk-b", [10.0, 0.0, 0.0], page=2),
        ]
    )

    reloaded = FAISSStore(
        snapshot_directory, dimension=DIMENSION, embedding_model=MODEL
    )
    results = reloaded.search([0.1, 0.0, 0.0], k=2)

    assert [result["chunk_id"] for result in results] == ["chunk-a", "chunk-b"]
    assert results[0]["text"] == "text for chunk-a"
    assert results[0]["metadata"]["labels"] == ["one", "two"]
    assert results[0]["metadata"]["nested"] == {"active": True, "score": 1.5}
    reloaded_record = reloaded.get_record("chunk-b")
    assert reloaded_record is not None
    assert reloaded_record["metadata"]["page_number"] == 2

    _, saved_manifest = manifest(snapshot_directory)
    assert saved_manifest["schema_version"] == 1
    assert saved_manifest["embedding_dimension"] == DIMENSION
    assert saved_manifest["embedding_model"] == MODEL
    assert len(saved_manifest["records"]) == reloaded.index.ntotal == 2


def test_snapshot_reloads_in_a_clean_process(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])

    script = (
        "import json; "
        "from src import vectorstore; "
        "FAISSStore=vectorstore.faiss.FAISSStore; "
        f"s=FAISSStore({str(snapshot_directory)!r}, dimension=3, "
        "embedding_model='test-embedding-model'); "
        "print(json.dumps(s.search([0.0,0.0,0.0], k=1)))"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)
    assert result[0]["chunk_id"] == "chunk-a"
    assert result[0]["metadata"]["document_id"] == "doc-a"


def test_empty_index_and_large_k_are_handled():
    store = FAISSStore(dimension=DIMENSION, embedding_model=MODEL)
    assert store.search([0.0, 0.0, 0.0], k=10) == []

    store.add_embedded_chunks([embedded_chunk("only", [1.0, 0.0, 0.0])])
    results = store.search([1.0, 0.0, 0.0], k=10)
    assert [result["chunk_id"] for result in results] == ["only"]


def test_query_and_added_vector_dimensions_are_validated():
    store = FAISSStore(dimension=DIMENSION, embedding_model=MODEL)
    with pytest.raises(DimensionMismatchError, match="Query embedding dimension"):
        store.search([0.0, 0.0], k=1)
    with pytest.raises(DimensionMismatchError, match="expected 3"):
        store.add_embedded_chunks([embedded_chunk("bad", [0.0, 0.0])])
    with pytest.raises(InvalidVectorRecordError, match="non-finite"):
        store.search([0.0, float("nan"), 0.0], k=1)
    with pytest.raises(InvalidVectorRecordError, match="numeric"):
        store.add_embedded_chunks([embedded_chunk("bad", [0.0, "not-a-number", 0.0])])


def test_duplicate_chunk_ids_are_rejected_before_mutation():
    store = FAISSStore(dimension=DIMENSION, embedding_model=MODEL)
    duplicate = embedded_chunk("same", [0.0, 0.0, 0.0])
    with pytest.raises(DuplicateChunkIDError):
        store.add_embedded_chunks([duplicate, duplicate])
    assert store.record_count == 0
    assert store.index.ntotal == 0


def test_store_rejects_chunks_outside_the_shared_schema():
    store = FAISSStore(dimension=DIMENSION, embedding_model=MODEL)
    invalid = embedded_chunk("invalid", [0.0, 0.0, 0.0])
    del invalid["metadata"]["source_type"]

    with pytest.raises(InvalidVectorRecordError, match="shared chunk schema"):
        store.add_embedded_chunks([invalid])

    assert store.record_count == 0


def test_incomplete_generation_is_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    generation = snapshot_directory / "snapshots" / "snapshot-incomplete"
    generation.mkdir(parents=True)
    (snapshot_directory / "CURRENT").write_text("snapshot-incomplete", encoding="utf-8")

    with pytest.raises(CorruptSnapshotError, match="incomplete"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)


def test_invalid_manifest_json_is_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])
    path, _ = manifest(snapshot_directory)
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(CorruptSnapshotError, match="Could not read"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)


def test_corrupted_index_bytes_are_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])
    (generation_directory(snapshot_directory) / "index.faiss").write_bytes(
        b"not-a-faiss-index"
    )

    with pytest.raises(CorruptSnapshotError, match="Could not read"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)


def test_index_and_metadata_count_mismatch_is_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])
    path, data = manifest(snapshot_directory)
    data["records"] = []
    write_manifest(path, data)

    with pytest.raises(CorruptSnapshotError, match="1 vectors.*0 records"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)


def test_corrupted_record_metadata_is_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])
    path, data = manifest(snapshot_directory)
    data["records"][0]["metadata"] = "not-an-object"
    write_manifest(path, data)

    with pytest.raises(CorruptSnapshotError, match="metadata is not a dictionary"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)


def test_snapshot_record_outside_shared_schema_is_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])
    path, data = manifest(snapshot_directory)
    del data["records"][0]["metadata"]["part_count"]
    write_manifest(path, data)

    with pytest.raises(CorruptSnapshotError, match="shared chunk schema"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)


def test_snapshot_dimension_and_model_mismatches_are_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])

    with pytest.raises(DimensionMismatchError, match="Snapshot dimension"):
        FAISSStore(snapshot_directory, dimension=4, embedding_model=MODEL)
    with pytest.raises(IncompatibleSnapshotError, match="embedding model"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model="other")


def test_snapshot_schema_mismatch_is_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])
    path, data = manifest(snapshot_directory)
    data["schema_version"] = 999
    write_manifest(path, data)

    with pytest.raises(IncompatibleSnapshotError, match="schema 999"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)


def test_interrupted_pointer_update_keeps_previous_snapshot(
    workspace_tmp_path, monkeypatch
):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks([embedded_chunk("chunk-a", [0.0, 0.0, 0.0])])
    original_replace = faiss_store_module.os.replace

    def fail_current_pointer(source, destination):
        if Path(destination).name == "CURRENT":
            raise OSError("simulated interrupted pointer update")
        return original_replace(source, destination)

    monkeypatch.setattr(faiss_store_module.os, "replace", fail_current_pointer)
    with pytest.raises(FAISSStoreError, match="simulated interrupted"):
        store.add_embedded_chunks([embedded_chunk("chunk-b", [1.0, 0.0, 0.0])])

    monkeypatch.setattr(faiss_store_module.os, "replace", original_replace)
    reloaded = FAISSStore(
        snapshot_directory, dimension=DIMENSION, embedding_model=MODEL
    )
    assert [record["chunk_id"] for record in reloaded.records] == ["chunk-a"]


def test_duplicate_ids_in_snapshot_are_rejected(workspace_tmp_path):
    snapshot_directory = workspace_tmp_path / "store"
    store = FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
    store.add_embedded_chunks(
        [
            embedded_chunk("chunk-a", [0.0, 0.0, 0.0]),
            embedded_chunk("chunk-b", [1.0, 0.0, 0.0]),
        ]
    )
    path, data = manifest(snapshot_directory)
    data["records"][1]["chunk_id"] = "chunk-a"
    write_manifest(path, data)

    with pytest.raises(DuplicateChunkIDError, match="chunk-a"):
        FAISSStore(snapshot_directory, dimension=DIMENSION, embedding_model=MODEL)
