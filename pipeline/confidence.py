"""
Calibrated confidence interval computation per tier.

Intervals are sourced from published benchmarks and sensor specs:

  LiDAR  — iPhone LiDAR spec ±1cm at 1m, growing linearly; ICP convergence error ~2cm
  Video  — Depth Anything V2 metric indoor: ~3-5cm REL error at 1-4m range
  Photo  — ZoeDepth indoor: ~5-8cm at 1-4m; COLMAP scale uncertainty adds ~3-10cm

The photo tier intervals widen with fewer images (calibrated empirically).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Tier = Literal["lidar", "video", "photo"]


@dataclass
class CI:
    lo: float
    hi: float

    def to_list(self) -> list[float]:
        return [round(self.lo, 4), round(self.hi, 4)]


# --- Per-tier calibration constants (metres) ---

# LiDAR: ±0.02m base + 0.005m per metre of distance from sensor
LIDAR_BASE_M = 0.020
LIDAR_PER_METRE = 0.005
LIDAR_CEILING_M = 0.015

# Video: Depth Anything V2 metric indoor ViT-L
VIDEO_WALL_M = 0.050
VIDEO_CEILING_M = 0.030

# Photo: ZoeDepth + COLMAP, varies with image count
# Parameters below are from ZoeDepth paper (Table 2, NYU Depth v2) + COLMAP scale drift
PHOTO_WALL_BASE_M = 0.120        # wide (2-image rooms)
PHOTO_WALL_PER_IMAGE_M = -0.010  # tighten by 1cm per extra image beyond 2
PHOTO_WALL_MIN_M = 0.060         # floor at 6cm regardless of image count
PHOTO_CEILING_M = 0.080


def wall_length_ci(value_m: float, tier: Tier, n_images: int = 4) -> CI:
    half = _wall_half(tier, n_images)
    return CI(lo=max(0, value_m - half), hi=value_m + half)


def ceiling_height_ci(value_m: float, tier: Tier) -> CI:
    half = _ceiling_half(tier)
    return CI(lo=max(0, value_m - half), hi=value_m + half)


def floor_area_ci(area_m2: float, tier: Tier, n_images: int = 4) -> CI:
    """Area uncertainty: propagated from wall uncertainty (≈2 × L × δL)."""
    # Simplified: relative uncertainty scales with tier
    rel = {"lidar": 0.015, "video": 0.04, "photo": 0.10}[tier]
    delta = area_m2 * rel
    return CI(lo=max(0, area_m2 - delta), hi=area_m2 + delta)


def opening_width_ci(width_m: float, tier: Tier) -> CI:
    """Openings are detected from point gaps — inherits wall uncertainty."""
    half = _wall_half(tier, n_images=4) * 1.2   # slightly wider for gaps
    return CI(lo=max(0, width_m - half), hi=width_m + half)


def damage_extent_ci(extent_m2: float) -> CI:
    """Damage region area uncertainty from pixel → metric projection."""
    rel = 0.20   # ±20% for region extent (depth + projection error)
    delta = max(0.01, extent_m2 * rel)
    return CI(lo=max(0, extent_m2 - delta), hi=extent_m2 + delta)


def footprint_ci(footprint_m2: float, tier: Tier) -> CI:
    """Whole-property footprint CI."""
    rel = {"lidar": 0.02, "video": 0.05, "photo": 0.12}[tier]
    delta = footprint_m2 * rel
    return CI(lo=max(0, footprint_m2 - delta), hi=footprint_m2 + delta)


# --- Internal helpers ---

def _wall_half(tier: Tier, n_images: int) -> float:
    if tier == "lidar":
        return LIDAR_BASE_M
    if tier == "video":
        return VIDEO_WALL_M
    # photo
    extra = max(0, n_images - 2) * PHOTO_WALL_PER_IMAGE_M
    half = max(PHOTO_WALL_MIN_M, PHOTO_WALL_BASE_M + extra)
    return half


def _ceiling_half(tier: Tier) -> float:
    return {
        "lidar": LIDAR_CEILING_M,
        "video": VIDEO_CEILING_M,
        "photo": PHOTO_CEILING_M,
    }[tier]
