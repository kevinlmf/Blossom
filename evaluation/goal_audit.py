"""
Goal satisfaction reporting utilities.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from goal import GoalDefinition, GoalSatisfactionSnapshot


class GoalAudit:
    """
    Generates persistent audit reports that describe how well a training run
    satisfied the personalized goal function.
    """

    def __init__(self, output_dir: str = "outputs/goal_audit"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_report(
        self,
        goal: GoalDefinition,
        snapshot: GoalSatisfactionSnapshot,
        regime: str,
        metadata: Dict[str, str] | None = None,
    ) -> Tuple[Path, Dict[str, float]]:
        payload = {
            "goal": goal.to_dict(),
            "regime": regime,
            "snapshot": snapshot.to_dict(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        if metadata:
            payload["metadata"] = dict(metadata)

        filename = f"{goal.name}_{regime}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.output_dir / filename

        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        return report_path, payload

