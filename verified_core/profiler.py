from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ProfileRecord:
    category: str
    op_name: str
    shape: str
    duration_ms: float
    tag: str
    ok: bool = True
    extra: str = ""


class VerifyProfiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._records: List[ProfileRecord] = []

    def add(
        self,
        category: str,
        op_name: str,
        duration_ms: float,
        tag: str,
        shape: str = "",
        ok: bool = True,
        extra: str = "",
    ) -> None:
        if not self.enabled:
            return
        self._records.append(
            ProfileRecord(
                category=category,
                op_name=op_name,
                shape=shape,
                duration_ms=float(duration_ms),
                tag=tag,
                ok=ok,
                extra=extra,
            )
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self._records])

    def summary_frame(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return df
        return (
            df.groupby(["category", "op_name"], as_index=False)["duration_ms"]
            .agg(["count", "mean", "max", "min"])
            .reset_index()
        )

    def export_json(self, detail_path: str, summary_path: Optional[str] = None) -> None:
        import json
        out = Path(detail_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([asdict(r) for r in self._records], f, indent=2)
        if summary_path is not None:
            with open(Path(summary_path), "w") as f:
                json.dump(self.summary_frame().to_dict("records"), f, indent=2)

    def export_csv(self, detail_path: str, summary_path: Optional[str] = None) -> None:
        df = self.to_frame()
        out_path = Path(detail_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        if summary_path is not None:
            self.summary_frame().to_csv(summary_path, index=False)

    def plot(self, output_path: str) -> None:
        if not self.enabled:
            return
        df = self.to_frame()
        if df.empty:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        agg = df.groupby(["category"], as_index=False)["duration_ms"].sum()
        plt.figure(figsize=(8, 4))
        plt.bar(agg["category"], agg["duration_ms"])
        plt.ylabel("Total ms")
        plt.title("ZImage Verify Runtime Breakdown")
        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
        plt.close()

    @property
    def records(self) -> List[ProfileRecord]:
        return self._records
