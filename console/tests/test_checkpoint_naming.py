"""Tests for the checkpoint stem helper and game slug.

The stem rule is documented in EXECPLAN_UNIVERSAL_RETRO_DQN.md (Decision Log
2026-05-07): ``{game_slug}_d{0|1}_u{0|1}_v{YYYYMMDDTHHMMSSZ}.pt`` (extension
added by the caller). This test pins the format down so future refactors can
catch regressions immediately.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.train import game_slug, make_checkpoint_stem


def test_game_slug_replaces_path_separators():
    assert game_slug("Sonic/Genesis") == "Sonic_Genesis"
    assert game_slug("foo bar") == "foo_bar"
    assert game_slug("MsPacMan-Nes-v0") == "MsPacMan-Nes-v0"


def test_game_slug_collapses_repeated_illegal_chars():
    assert game_slug("a///b") == "a_b"
    assert game_slug("a:::b") == "a_b"


def test_game_slug_rejects_empty():
    with pytest.raises(ValueError):
        game_slug("")


def test_make_checkpoint_stem_format():
    when = datetime(2026, 5, 7, 14, 30, 22, tzinfo=timezone.utc)
    assert (
        make_checkpoint_stem("Airstriker-Genesis", False, False, when)
        == "Airstriker-Genesis_d0_u0_v20260507T143022Z"
    )
    assert (
        make_checkpoint_stem("Airstriker-Genesis", True, True, when)
        == "Airstriker-Genesis_d1_u1_v20260507T143022Z"
    )


def test_make_checkpoint_stem_truncates_microseconds_and_uses_utc():
    when_local = datetime(2026, 5, 7, 7, 30, 22, 999_999)
    # Naive datetimes are treated as UTC by the helper.
    stem_naive = make_checkpoint_stem("g", True, False, when_local)
    assert stem_naive == "g_d1_u0_v20260507T073022Z"

    # An explicit US/Pacific (-07:00) offset is converted to UTC.
    from datetime import timedelta

    pacific = timezone(timedelta(hours=-7))
    when_pac = datetime(2026, 5, 7, 7, 30, 22, tzinfo=pacific)
    stem_pac = make_checkpoint_stem("g", False, True, when_pac)
    assert stem_pac == "g_d0_u1_v20260507T143022Z"
