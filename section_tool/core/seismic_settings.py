"""Phase 5 — Per-section seismic display settings."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeismicDisplaySettings:
    colormap:           str   = "gray_r"
    gain:               float = 1.0
    clip_percentile:    float = 99.0
    opacity:            float = 1.0
    show_wiggle:        bool  = False
    wiggle_fill:        bool  = True
    # TWT-to-depth stretch (FIX 5)
    stretch_mode:       str   = "linear"    # "linear" | "native_twt"
    constant_velocity:  float = 2000.0      # m/s, used for linear stretch

    def to_dict(self) -> dict:
        return {
            "colormap":          self.colormap,
            "gain":              self.gain,
            "clip_percentile":   self.clip_percentile,
            "opacity":           self.opacity,
            "show_wiggle":       self.show_wiggle,
            "wiggle_fill":       self.wiggle_fill,
            "stretch_mode":      self.stretch_mode,
            "constant_velocity": self.constant_velocity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SeismicDisplaySettings":
        return cls(
            colormap=d.get("colormap", "gray_r"),
            gain=d.get("gain", 1.0),
            clip_percentile=d.get("clip_percentile", 99.0),
            opacity=d.get("opacity", 1.0),
            show_wiggle=d.get("show_wiggle", False),
            wiggle_fill=d.get("wiggle_fill", True),
            stretch_mode=d.get("stretch_mode", "linear"),
            constant_velocity=d.get("constant_velocity", 2000.0),
        )
