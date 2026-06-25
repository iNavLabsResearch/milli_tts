"""Continuously back up training checkpoints to a Hugging Face model repo.

So a Kaggle disconnect never loses progress: every periodic save (and the final
one) is uploaded to ``<owner>/<checkpoint_repo>`` under a **per-run folder**
named from an incrementing index + the launch timestamp, e.g.
``run003-20260625-070000/``. Within a run we keep a rolling ``latest.pt`` (for
resume) plus ``final.pt`` at the end.

Uploads run on a background thread so they never block the training step; if a
push is still in flight when the next one is due, it's skipped (the next save
catches up). Only rank 0 uploads. All network calls are best-effort — any
failure is logged and training continues.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

from milli_tts.core.logger import get_logger

log = get_logger("training.hf_sync")


class HFCheckpointSync:
    def __init__(self, *, repo: str, token: Optional[str], enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.repo_id: Optional[str] = None
        self.run_id: str = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        self._token = token
        self._api = None
        self._thread: Optional[threading.Thread] = None
        if not self.enabled:
            return
        try:
            from huggingface_hub import HfApi, create_repo

            self._api = HfApi(token=token)
            info = create_repo(repo, repo_type="model", private=True,
                               exist_ok=True, token=token)
            self.repo_id = info.repo_id
            self.run_id = self._make_run_id()
            log.info("HF checkpoint backup ON: repo=%s run=%s (uploads in "
                     "background, rank 0 only).", self.repo_id, self.run_id)
        except Exception as exc:  # pragma: no cover - network/env dependent
            log.warning("HF checkpoint backup disabled (setup failed: %s). "
                        "Training continues with local checkpoints only.", exc)
            self.enabled = False

    # ------------------------------------------------------------------ #
    def _make_run_id(self) -> str:
        """``run{NNN}-{timestamp}``: index increments per existing run folder."""
        idx = 1
        try:
            files = self._api.list_repo_files(self.repo_id, repo_type="model")
            runs = {f.split("/", 1)[0] for f in files if f.startswith("run")}
            idx = len(runs) + 1
        except Exception:  # empty/new repo or transient error -> start at 1
            pass
        return datetime.now().strftime(f"run{idx:03d}-%Y%m%d-%H%M%S")

    def _upload(self, local_path: str, name: str, *, blocking: bool) -> None:
        path_in_repo = f"{self.run_id}/{name}"

        def _do() -> None:
            try:
                self._api.upload_file(
                    path_or_fileobj=local_path, path_in_repo=path_in_repo,
                    repo_id=self.repo_id, repo_type="model",
                    commit_message=f"{name} @ {self.run_id}")
                log.info("Pushed checkpoint -> hf://%s/%s", self.repo_id,
                         path_in_repo)
            except Exception as exc:  # pragma: no cover
                log.warning("HF checkpoint push failed (%s): %s", name, exc)

        if blocking:
            _do()
        elif self._thread is not None and self._thread.is_alive():
            log.info("Skipping HF push of %s — previous upload still running.",
                     name)
        else:
            self._thread = threading.Thread(target=_do, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ #
    def push_latest(self, local_path: str) -> None:
        """Rolling resume point, uploaded in the background (non-blocking)."""
        if self.enabled:
            self._upload(local_path, "latest.pt", blocking=False)

    def push_final(self, local_path: str) -> None:
        """End-of-run checkpoint — wait for any in-flight push, then upload."""
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=900)
        self._upload(local_path, "final.pt", blocking=True)
