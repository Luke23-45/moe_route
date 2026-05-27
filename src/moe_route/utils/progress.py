from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from tqdm.std import tqdm


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = True
    dynamic_ncols: bool = True
    leave: bool = True
    mininterval: float = 0.5


def progress_cfg(cfg) -> ProgressConfig:
    node = cfg.trainer.get("progress", {})
    return ProgressConfig(
        enabled=bool(node.get("enabled", True)),
        dynamic_ncols=bool(node.get("dynamic_ncols", True)),
        leave=bool(node.get("leave", True)),
        mininterval=float(node.get("mininterval", 0.5)),
    )


def training_bar(cfg, epoch: int, total: int, initial: int = 0):
    progress = progress_cfg(cfg)
    if not progress.enabled:
        return _noop_bar()

    router_name = "dense" if not bool(cfg.model.moe.enabled) else str(cfg.router.name)
    desc = (
        f"{cfg.experiment.name} | data={cfg.data.name} | model={cfg.model.name} "
        f"| router={router_name} | epoch={epoch}"
    )

    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        unit="step",
        dynamic_ncols=progress.dynamic_ncols,
        leave=progress.leave,
        mininterval=progress.mininterval,
        position=0,
        file=_single_line_stream(sys.stderr),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


class _single_line_stream:
    """
    Keeps tqdm on one visual line.
    - carriage returns rewrite the same line
    - newline is emitted only when tqdm closes
    """

    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True

    def write(self, s: str) -> None:
        if not s:
            return

        # tqdm emits lots of '\r'. We keep that behavior, but never turn it into spammy '\n'.
        # ANSI clear-line helps many terminals/log viewers.
        if "\r" in s:
            parts = s.split("\r")
            for i, part in enumerate(parts):
                if i > 0:
                    self._stream.write("\r")
                    self._stream.write("\x1b[2K")  # clear current line
                if part:
                    self._stream.write(part)
            self._stream.flush()
            self._at_line_start = False
            return

        self._stream.write(s)
        self._stream.flush()
        self._at_line_start = s.endswith("\n")

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


class _noop_bar:
    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()