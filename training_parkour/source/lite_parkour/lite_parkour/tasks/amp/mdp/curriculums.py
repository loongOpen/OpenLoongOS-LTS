"""Curriculum terms for the AMP locomotion environments."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Runtime classes stay behind TYPE_CHECKING, per the pattern the sibling mdp modules
    # follow. Importing isaaclab.assets / isaaclab.terrains at module scope pulls USD/pxr
    # into the process before AppLauncher initializes it, and Kit then aborts on startup
    # with "free(): invalid pointer" -- see the note in this package's __init__.
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter

__all__ = ["terrain_levels_by_type"]

#: Cache of the column-index -> sub-terrain-name map, keyed by terrain generator config id.
_COLUMN_NAMES: dict[int, list[str]] = {}


def _column_sub_terrain_names(terrain: TerrainImporter) -> list[str]:
    """Which sub-terrain each grid column holds.

    ``terrain_types`` is a *column* index, not a sub-terrain index: the generator hands
    each sub-terrain a contiguous block of columns sized by its normalized proportion, so
    with ``num_cols=20`` and eight equally weighted sub-terrains the blocks come out two
    or three columns wide. This reproduces the assignment
    ``TerrainGenerator._generate_curriculum_terrains`` makes so the log can be labelled by
    name instead of by column number.
    """
    cfg = terrain.cfg.terrain_generator
    key = id(cfg)
    if key not in _COLUMN_NAMES:
        names = list(cfg.sub_terrains.keys())
        proportions = np.array([sub.proportion for sub in cfg.sub_terrains.values()], dtype=float)
        proportions /= proportions.sum()
        cumulative = np.cumsum(proportions)
        _COLUMN_NAMES[key] = [
            names[int(np.min(np.where(col / cfg.num_cols + 0.001 < cumulative)[0]))] for col in range(cfg.num_cols)
        ]
    return _COLUMN_NAMES[key]


def terrain_levels_by_type(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """Report the mean terrain level of each sub-terrain. Read-only.

    This term promotes nothing -- it is purely a logging companion to ``terrain_levels_vel``,
    which does the promotion and stays responsible for the ``Curriculum/terrain_levels``
    scalar. Splitting them keeps that scalar's tag byte-identical to earlier runs, so old
    and new curves overlay; a term returning a dict would have been relabelled
    ``Curriculum/terrain_levels/mean`` by ``CurriculumManager.reset``.

    Why the breakdown is worth logging: a global mean cannot distinguish "everything sits
    at 5" from "eight sub-terrains reached 6 and two are pinned at 0" -- the same number,
    opposite diagnoses. Since ``terrain_types`` fixes each environment to one column for
    its whole life, the per-column breakdown separates them exactly.

    Declare it *after* ``terrain_levels`` in the curriculum cfg: ``CurriculumManager.compute``
    runs terms in ``__dict__`` order, so this then reads levels that already include this
    step's promotions.

    Logged as ``Curriculum/terrain_levels_by_type/<sub_terrain_name>``.
    """
    del env_ids  # reported over all environments, like the stock scalar
    terrain: TerrainImporter = env.scene.terrain
    num_cols = terrain.cfg.terrain_generator.num_cols

    # One bincount pass and a single device->host transfer. The obvious loop -- mask per
    # column, ``mask.any()``, ``levels[mask].mean()``, then ``.item()`` on each entry --
    # costs ~30 GPU syncs, and _reset_idx runs this on nearly every step at 4096 envs.
    # Measured at 4096 envs / 20 columns: 894 us per call that way, 90 us this way.
    levels = terrain.terrain_levels.float()
    counts = torch.bincount(terrain.terrain_types, minlength=num_cols).float()
    sums = torch.bincount(terrain.terrain_types, weights=levels, minlength=num_cols)
    packed = torch.cat((sums, counts)).cpu().tolist()
    column_sums, column_counts = packed[:num_cols], packed[num_cols:]

    # Several columns can hold the same sub-terrain; pool them into one population mean
    # rather than averaging the column means, which would misweight unequal columns.
    totals: dict[str, list[float]] = {}
    for column, name in enumerate(_column_sub_terrain_names(terrain)):
        entry = totals.setdefault(name, [0.0, 0.0])
        entry[0] += column_sums[column]
        entry[1] += column_counts[column]
    return {name: total / count for name, (total, count) in totals.items() if count > 0.0}
