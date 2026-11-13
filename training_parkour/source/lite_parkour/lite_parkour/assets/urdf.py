"""URDF spawning helpers for the Lite articulation."""

import isaaclab.sim as sim_utils


def _activate_all_nested_rigid_body_contacts(prim_path: str) -> None:
    """Enable contact reports on every rigid body in a nested URDF articulation.

    Isaac Lab v3's generic contact activation stops descending as soon as it
    encounters a rigid body.  The v3 URDF importer represents a torso-root
    articulation as nested rigid-body prims, so the generic traversal only
    enables the torso contact reporter.  Starting activation separately at
    each rigid-body prim preserves the normal schema setup while covering the
    pelvis, feet, hands, and all other links.
    """
    # Import USD schemas lazily: importing ``pxr`` before Isaac Sim launches
    # causes Kit plug-ins to initialize against the wrong runtime.
    from pxr import UsdPhysics

    stage = sim_utils.get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    rigid_body_paths = []
    prims = [root_prim]
    while prims:
        prim = prims.pop(0)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_paths.append(prim.GetPath().pathString)
        prims.extend(prim.GetChildren())

    if not rigid_body_paths:
        raise ValueError(f"No rigid bodies found below '{prim_path}'.")
    for rigid_body_path in rigid_body_paths:
        sim_utils.activate_contact_sensors(rigid_body_path)


def _replace_cylinder_collisions(prim_path: str, preserve_extent: bool = True) -> None:
    """Convert cylinder contacts to capsules; preserve axial extent when possible.

    Short cylinders become spheres when their diameter exceeds their length.
    """
    from pxr import UsdGeom, UsdPhysics

    stage = sim_utils.get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    cylinders = []
    prims = [root_prim]
    while prims:
        prim = prims.pop(0)
        if prim.GetTypeName() == "Cylinder" and prim.HasAPI(UsdPhysics.CollisionAPI):
            cylinders.append(prim)
        prims.extend(prim.GetChildren())

    for prim in cylinders:
        cylinder = UsdGeom.Cylinder(prim)
        radius = cylinder.GetRadiusAttr().Get()
        height = cylinder.GetHeightAttr().Get()
        axis = cylinder.GetAxisAttr().Get() or "Z"
        path = prim.GetPath().pathString
        # ``RemovePrim`` keeps the authored xform opinions in the layer, so the
        # redefined prim inherits the ops -- capture the values and write them
        # back rather than adding a second set.
        saved_ops = [(op.GetOpName(), op.Get()) for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
        purpose_attr = UsdGeom.Imageable(prim).GetPurposeAttr()
        purpose = purpose_attr.Get() if purpose_attr.HasAuthoredValue() else None

        stage.RemovePrim(path)

        shape = UsdGeom.Capsule.Define(stage, path)
        shape.GetRadiusAttr().Set(radius)
        # Clamps to 0 for squat cylinders, which makes the capsule a sphere.
        shape.GetHeightAttr().Set(max(height - 2.0 * radius, 0.0) if preserve_extent else height)
        shape.GetAxisAttr().Set(axis)

        existing_ops = {op.GetOpName(): op for op in UsdGeom.Xformable(shape).GetOrderedXformOps()}
        for op_name, value in saved_ops:
            if value is not None and op_name in existing_ops:
                existing_ops[op_name].Set(value)
        if purpose is not None:
            UsdGeom.Imageable(shape).CreatePurposeAttr().Set(purpose)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())


def spawn_nested_contact_urdf(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a URDF and register contacts for its nested rigid-body links."""
    # Import the spawner utilities lazily so task registration does not import
    # USD modules before Isaac Sim's SimulationApp has started.
    from isaaclab.sim.spawners.from_files import spawn_from_urdf
    from isaaclab.sim.utils import clone

    def spawn_source(
        source_prim_path: str,
        source_cfg: sim_utils.UrdfFileCfg,
        translation=None,
        orientation=None,
        **source_kwargs,
    ):
        # Call the undecorated Isaac Lab spawner. The dynamic ``clone`` wrapper
        # below will copy the fully prepared source, including every contact
        # schema, to all remaining environments.
        source_prim = spawn_from_urdf.__wrapped__(
            source_prim_path,
            source_cfg,
            translation=translation,
            orientation=orientation,
            **source_kwargs,
        )
        # Must run on the source prim, before the cloner copies it: the
        # conversion then reaches every environment for the price of one.
        _replace_cylinder_collisions(source_prim_path)
        if source_cfg.activate_contact_sensors:
            _activate_all_nested_rigid_body_contacts(source_prim_path)
        return source_prim

    return clone(spawn_source)(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
