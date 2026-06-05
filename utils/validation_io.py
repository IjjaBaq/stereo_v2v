"""Shared I/O helpers for validation result files.

Validation scripts write a single ``validation_results.json`` per
method/sequence. Running validation on a subset of samples must not clobber
results already on disk — instead, new per-item entries are merged into the
existing list (upsert by ID) so aggregated metrics can be recomputed over the
full accumulated set.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_samples(
    results_path: Path,
    new_items: list[dict],
    id_key: str,
    list_key: str,
) -> list[dict]:
    """Upsert per-item validation entries into an existing results file.

    Reads ``list_key`` from ``results_path`` (if the file exists), replaces any
    entry whose ``id_key`` matches a new item, appends the rest, and returns the
    merged list sorted by ``id_key``. A missing file or missing ``list_key`` is
    treated as an empty existing list.

    Args:
        results_path: Path to the validation_results.json file.
        new_items: Newly validated entries, each containing ``id_key``.
        id_key: Per-item identity field (e.g. 'sample_id', 'frame_id').
        list_key: Top-level key holding the item list (e.g. 'samples', 'frames').

    Returns:
        Merged list of entries, sorted by ``id_key``.
    """
    merged: dict = {}

    if results_path.exists():
        try:
            with open(results_path) as f:
                existing = json.load(f)
            for item in existing.get(list_key, []):
                if id_key in item:
                    merged[item[id_key]] = item
            logger.info(
                "Merging into existing results (%d prior entries) → %s",
                len(merged), results_path,
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Could not read existing results at %s (%s) — overwriting.",
                results_path, e,
            )
            merged = {}

    for item in new_items:
        if id_key in item:
            merged[item[id_key]] = item

    return [merged[k] for k in sorted(merged)]
