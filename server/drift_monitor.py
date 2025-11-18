from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class FeatureStats:
    """Online mean/variance tracker using Welford's algorithm."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        v = self.variance
        return v ** 0.5 if v > 0 else 0.0


@dataclass
class DriftMonitor:
    """
    Very simple drift monitor:
    - Keeps online stats of live data
    - Compares against baseline means/stds
    - Returns z-scores + boolean drift flag
    """
    baseline_means: Dict[str, float]
    baseline_stds: Dict[str, float]
    threshold: float = 3.0  # z-score threshold
    live_stats: Dict[str, FeatureStats] = field(default_factory=dict)

    def update_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update internal stats from a single feature row and return drift info:
        {
          "per_feature": { "midprice": 0.7, ... },
          "overall": 1.2,
          "is_drift": False
        }
        """
        per_feature_z = {}

        for feat, base_mean in self.baseline_means.items():
            if feat not in row:
                continue

            value = float(row[feat])
            stats = self.live_stats.setdefault(feat, FeatureStats())
            stats.update(value)

            base_std = self.baseline_stds.get(feat, 0.0) or 1e-9
            z = abs((stats.mean - base_mean) / base_std)
            per_feature_z[feat] = z

        overall = max(per_feature_z.values()) if per_feature_z else 0.0
        is_drift = overall > self.threshold

        return {
            "per_feature": per_feature_z,
            "overall": overall,
            "is_drift": is_drift,
        }
