# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain generation sub-package for lite_parkour.

This module bundles the Perlin-augmented sub-terrains ported from instinctlab
(height-field and trimesh families) together with the custom terrain generator
that supports per-sub-terrain noise and surrounding walls. Official IsaacLab
sub-terrains remain importable from ``isaaclab.terrains`` and are combined with
these in the task terrain configuration.

Lazy exports (IMPORTANT)
------------------------
Exports are declared in the sibling ``__init__.pyi`` stub and attached lazily via
``lazy_export()`` (same mechanism as official ``isaaclab.terrains``). Importing
``lite_parkour.terrains`` must stay side-effect-free — it must NOT eagerly import
``isaaclab.terrains.TerrainGenerator`` (pulls in ``pxr``/USD native bindings) nor
``torch`` (via the trimesh sub-package). Those heavy native libraries loading
*before* Isaac Sim's ``SimulationApp`` boots corrupts the native heap and crashes
kit during its plugin ``dlopen`` phase (``libusd_tf.so`` abort in ``TfEnum::_AddName``).
Because ``lite_parkour.tasks`` (and thus this package, via the terrain config) is
imported *before* the app launches in the run scripts, eager imports here are exactly
what triggers that startup crash. ``lazy_export`` resolves each name to its submodule
only on first attribute access, so the light height-field cfgs (numpy/scipy) never
drag in ``pxr``/``torch``.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
