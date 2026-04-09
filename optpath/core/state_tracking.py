"""Excited-state tracking logic."""

from __future__ import annotations

from dataclasses import dataclass

from optpath.engines.results import ImageResult, TrackedState


@dataclass(slots=True)
class StateTracker:
    enabled: bool = True

    def update(
        self,
        previous_results: list[ImageResult] | None,
        current_results: list[ImageResult],
    ) -> list[TrackedState]:
        if not self.enabled:
            return [
                TrackedState(
                    image_index=result.image_index,
                    selected_root=result.selected_root,
                    state_label=result.state_label,
                    warnings=list(result.warnings),
                    metadata={"selected_root": result.selected_root},
                )
                for result in current_results
            ]
        previous_map = {result.image_index: result for result in (previous_results or [])}
        tracked: list[TrackedState] = []
        for result in current_results:
            warnings = list(result.warnings)
            previous = previous_map.get(result.image_index)
            if previous and previous.selected_root is not None and result.selected_root is not None:
                if previous.selected_root != result.selected_root:
                    warnings.append(
                        f"selected root changed from {previous.selected_root} to {result.selected_root}"
                    )
            tracked.append(
                TrackedState(
                    image_index=result.image_index,
                    selected_root=result.selected_root,
                    state_label=result.state_label,
                    warnings=warnings,
                    metadata={
                        "selected_root": result.selected_root,
                        "available_roots": result.available_roots,
                    },
                )
            )
        return tracked

