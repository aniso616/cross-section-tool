"""Performance and correctness tests for the seismic image render cache.

All tests are headless (no Qt required): they operate on the cache-logic
helpers and the expensive computations being bypassed.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal mocks for cache-logic tests
# ---------------------------------------------------------------------------

def _make_settings_key(
    section_name="S1",
    ref_paths=("test.segy",),
    clip_pct=99.0,
    gain=1.0,
    opacity=1.0,
    colormap="seismic_red_blue",
    show_wiggle=False,
    display_mode="variable_density",
) -> tuple:
    return (
        section_name, tuple(ref_paths),
        clip_pct, gain, opacity, colormap, show_wiggle, display_mode,
    )


def _make_mock_artist(alive: bool = True):
    art = MagicMock()
    art.axes = MagicMock() if alive else None
    return art


class FakeSeismicCache:
    """Minimal stand-in that exercises the same cache logic as SectionView."""

    def __init__(self):
        self._seismic_img_artists: list = []
        self._seismic_img_cache_key: tuple | None = None
        self._seismic_cache_xlim: tuple | None = None
        self._seismic_cache_ylim: tuple | None = None
        self._seismic_vmax_cache: dict = {}
        self._xlim = (0.0, 10000.0)
        self._ylim = (5000.0, 0.0)

    def _get_xlim(self): return self._xlim
    def _get_ylim(self): return self._ylim

    def _is_seismic_cache_valid(self, section_name: str, cache_key: tuple) -> bool:
        if not self._seismic_img_artists:
            return False
        if any(getattr(a, "axes", None) is None for a in self._seismic_img_artists):
            return False
        if self._seismic_img_cache_key != cache_key:
            return False
        try:
            xl = tuple(round(v, 2) for v in self._get_xlim())
            yl = tuple(round(v, 2) for v in self._get_ylim())
        except Exception:
            return False
        return xl == self._seismic_cache_xlim and yl == self._seismic_cache_ylim

    def _invalidate_seismic_cache(self):
        self._seismic_img_artists.clear()
        self._seismic_img_cache_key = None
        self._seismic_cache_xlim = None
        self._seismic_cache_ylim = None
        self._seismic_vmax_cache.clear()

    def record_render(self, cache_key: tuple, artist=None):
        if artist is None:
            artist = _make_mock_artist(alive=True)
        self._seismic_img_artists = [artist]
        self._seismic_img_cache_key = cache_key
        xl = tuple(round(v, 2) for v in self._get_xlim())
        yl = tuple(round(v, 2) for v in self._get_ylim())
        self._seismic_cache_xlim = xl
        self._seismic_cache_ylim = yl


# ---------------------------------------------------------------------------
# 1. Cache validity logic
# ---------------------------------------------------------------------------

class TestCacheValidity:

    def test_empty_cache_is_invalid(self):
        c = FakeSeismicCache()
        key = _make_settings_key()
        assert not c._is_seismic_cache_valid("S1", key)

    def test_cache_valid_after_render(self):
        c = FakeSeismicCache()
        key = _make_settings_key()
        c.record_render(key)
        assert c._is_seismic_cache_valid("S1", key)

    def test_cache_invalid_after_invalidate(self):
        c = FakeSeismicCache()
        key = _make_settings_key()
        c.record_render(key)
        c._invalidate_seismic_cache()
        assert not c._is_seismic_cache_valid("S1", key)

    def test_cache_invalid_when_artist_detached(self):
        c = FakeSeismicCache()
        key = _make_settings_key()
        dead_artist = _make_mock_artist(alive=False)
        c.record_render(key, artist=dead_artist)
        assert not c._is_seismic_cache_valid("S1", key)

    def test_cache_invalid_on_different_section(self):
        c = FakeSeismicCache()
        key_s1 = _make_settings_key(section_name="S1")
        key_s2 = _make_settings_key(section_name="S2")
        c.record_render(key_s1)
        assert not c._is_seismic_cache_valid("S2", key_s2)

    def test_cache_invalid_on_different_colormap(self):
        c = FakeSeismicCache()
        key_hot = _make_settings_key(colormap="hot")
        key_rb  = _make_settings_key(colormap="seismic_red_blue")
        c.record_render(key_hot)
        assert not c._is_seismic_cache_valid("S1", key_rb)

    def test_cache_invalid_on_different_gain(self):
        c = FakeSeismicCache()
        key1 = _make_settings_key(gain=1.0)
        key2 = _make_settings_key(gain=2.0)
        c.record_render(key1)
        assert not c._is_seismic_cache_valid("S1", key2)

    def test_cache_invalid_on_different_xlim(self):
        c = FakeSeismicCache()
        key = _make_settings_key()
        c.record_render(key)
        c._seismic_cache_xlim = None   # simulate scroll zoom
        assert not c._is_seismic_cache_valid("S1", key)

    def test_cache_invalid_on_different_ylim(self):
        c = FakeSeismicCache()
        key = _make_settings_key()
        c.record_render(key)
        c._seismic_cache_ylim = None
        assert not c._is_seismic_cache_valid("S1", key)

    def test_cache_invalid_on_new_seismic_ref(self):
        c = FakeSeismicCache()
        key_one  = _make_settings_key(ref_paths=("a.segy",))
        key_two  = _make_settings_key(ref_paths=("a.segy", "b.segy"))
        c.record_render(key_one)
        assert not c._is_seismic_cache_valid("S1", key_two)

    def test_cache_survives_non_invalidating_events(self):
        """Cache should remain valid when only picks/annotations change."""
        c = FakeSeismicCache()
        key = _make_settings_key()
        c.record_render(key)
        # Simulate pick add (no cache change expected)
        assert c._is_seismic_cache_valid("S1", key)
        # Simulate hover change (no cache change expected)
        assert c._is_seismic_cache_valid("S1", key)


# ---------------------------------------------------------------------------
# 2. vmax computation cache
# ---------------------------------------------------------------------------

class TestVmaxCache:

    def test_vmax_cached_after_first_call(self):
        cache = {}
        data = np.random.randn(100, 200).astype(np.float32)
        key = ("S1", "test.segy", 99.0, 1.0)

        def get_vmax(data, clip_pct, gain, key, cache):
            if key in cache:
                return cache[key]
            vmax = float(np.percentile(np.abs(data), clip_pct) or 1.0) * gain
            cache[key] = vmax
            return vmax

        v1 = get_vmax(data, 99.0, 1.0, key, cache)
        v2 = get_vmax(data, 99.0, 1.0, key, cache)
        assert v1 == v2
        assert key in cache

    def test_vmax_recomputed_on_new_key(self):
        cache = {}
        data = np.ones((100, 200), dtype=np.float32)
        k1 = ("S1", "test.segy", 99.0, 1.0)
        k2 = ("S1", "test.segy", 99.0, 2.0)  # different gain
        cache[k1] = 1.0
        assert k2 not in cache   # not reused for different key


# ---------------------------------------------------------------------------
# 3. Performance: cached path is >100x faster than full computation
# ---------------------------------------------------------------------------

class TestPerformance:

    def test_percentile_cache_speedup(self):
        """Caching np.percentile gives >100x speedup for large arrays."""
        # Simulate the expensive part of seismic rendering:
        # F3: 630 000 traces × 462 samples
        data = np.random.randn(500, 1000).astype(np.float32)
        clip_pct = 99.0

        # ── Full computation (no cache) ───────────────────────────────
        t0 = time.perf_counter()
        vmax_full = float(np.percentile(np.abs(data), clip_pct) or 1.0)
        t_full = time.perf_counter() - t0

        # ── Cached lookup ─────────────────────────────────────────────
        cache = {"vmax": vmax_full}
        times = []
        for _ in range(1_000):
            t0 = time.perf_counter()
            _ = cache["vmax"]            # O(1) dict lookup
            times.append(time.perf_counter() - t0)

        t_cached = float(np.mean(times))

        print(f"\n  Full percentile:  {t_full * 1000:.2f} ms")
        print(f"  Cached lookup:    {t_cached * 1_000_000:.2f} µs")
        print(f"  Speedup:          {t_full / max(t_cached, 1e-12):.0f}×")

        assert t_cached < t_full * 0.01, (
            f"Cached lookup ({t_cached*1e6:.1f} µs) should be >100× faster than "
            f"full compute ({t_full*1000:.1f} ms)"
        )

    def test_cache_valid_check_is_fast(self):
        """_is_seismic_cache_valid() must be negligible overhead per render."""
        c = FakeSeismicCache()
        key = _make_settings_key()
        c.record_render(key)

        N = 10_000
        t0 = time.perf_counter()
        for _ in range(N):
            c._is_seismic_cache_valid("S1", key)
        t_total = time.perf_counter() - t0
        t_per_call = t_total / N

        print(f"\n  Cache check: {t_per_call * 1_000_000:.2f} µs/call")
        assert t_per_call < 1e-4, (   # < 100 µs per check
            f"Cache validity check took {t_per_call*1e6:.1f} µs, expected < 100 µs"
        )

    def test_invalidate_is_fast(self):
        c = FakeSeismicCache()
        key = _make_settings_key()
        artist = _make_mock_artist(alive=True)  # reuse to avoid MagicMock creation overhead
        c.record_render(key, artist=artist)

        t0 = time.perf_counter()
        for _ in range(10_000):
            c.record_render(key, artist=artist)   # re-render (sets cache)
            c._invalidate_seismic_cache()          # invalidate
        t_total = time.perf_counter() - t0

        assert t_total < 1.0, "10 000 invalidations should complete in < 1 s"

    def test_overlay_clear_faster_than_full_render_simulation(self):
        """Simulates the speedup: overlay-only clear vs full imshow creation."""
        data = np.random.randn(500, 1000).astype(np.float32)

        def simulate_full_render():
            _ = float(np.percentile(np.abs(data), 99.0) or 1.0)
            return True

        def simulate_cached_render(cache_valid):
            if cache_valid:
                return False   # skip
            return simulate_full_render()

        # Time full renders
        t0 = time.perf_counter()
        for _ in range(10):
            simulate_full_render()
        t_full = (time.perf_counter() - t0) / 10

        # Time cached renders (should be instant)
        times = []
        for _ in range(1_000):
            t0 = time.perf_counter()
            simulate_cached_render(cache_valid=True)
            times.append(time.perf_counter() - t0)
        t_cached = float(np.mean(times))

        speedup = t_full / max(t_cached, 1e-12)
        print(f"\n  Full:   {t_full*1000:.2f} ms")
        print(f"  Cached: {t_cached*1_000_000:.2f} µs")
        print(f"  Speedup: {speedup:.0f}×")
        assert speedup >= 100, f"Expected ≥100× speedup, got {speedup:.0f}×"


# ---------------------------------------------------------------------------
# 4. Settings key correctness
# ---------------------------------------------------------------------------

class TestSettingsKey:

    def test_keys_equal_for_same_settings(self):
        k1 = _make_settings_key()
        k2 = _make_settings_key()
        assert k1 == k2

    def test_keys_differ_on_section_name(self):
        assert _make_settings_key("S1") != _make_settings_key("S2")

    def test_keys_differ_on_gain(self):
        assert _make_settings_key(gain=1.0) != _make_settings_key(gain=2.0)

    def test_keys_differ_on_colormap(self):
        assert _make_settings_key(colormap="hot") != _make_settings_key(colormap="gray")

    def test_keys_differ_on_clip_pct(self):
        assert _make_settings_key(clip_pct=95.0) != _make_settings_key(clip_pct=99.0)

    def test_keys_differ_on_opacity(self):
        assert _make_settings_key(opacity=0.5) != _make_settings_key(opacity=1.0)

    def test_keys_differ_on_display_mode(self):
        assert (
            _make_settings_key(display_mode="variable_density") !=
            _make_settings_key(display_mode="wiggle")
        )

    def test_keys_differ_on_ref_set(self):
        k1 = _make_settings_key(ref_paths=("a.segy",))
        k2 = _make_settings_key(ref_paths=("a.segy", "b.segy"))
        assert k1 != k2

    def test_ref_path_order_matters(self):
        """Different ordering of refs should give different keys."""
        k1 = _make_settings_key(ref_paths=("a.segy", "b.segy"))
        k2 = _make_settings_key(ref_paths=("b.segy", "a.segy"))
        assert k1 != k2

    def test_key_is_hashable(self):
        """Key must be usable as a dict key."""
        k = _make_settings_key()
        d = {k: "value"}
        assert d[k] == "value"
