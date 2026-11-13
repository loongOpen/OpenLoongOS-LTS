# SPDX-License-Identifier: MIT
"""Walking-style visual assets without changing physics or policy geometry."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


def _scene():
    return ET.parse(Path(__file__).resolve().parents[2] / "runtime/model/scene.xml").getroot()


def _floor_style(scene):
    """Read the material actually assigned to walking's included ground geom."""
    model_root = Path(__file__).resolve().parents[2] / "runtime/model"
    documents = [scene]
    documents.extend(ET.parse(model_root / item.get("file")).getroot() for item in scene.findall("include"))
    ground = next(geom for document in documents for geom in document.findall("worldbody/geom")
                  if geom.get("name") == "ground" and geom.get("type") == "plane")
    material = next(item for document in documents for item in document.findall("asset/material")
                    if item.get("name") == ground.get("material"))
    texture = next(item for document in documents for item in document.findall("asset/texture")
                   if item.get("name") == material.get("texture"))
    return ground, material, texture


def _numbers(value):
    return [float(x) for x in value.split()]


def add_native_assets(spec):
    """Add the walking sky and floor material before compiling the dance model."""
    import mujoco
    scene = _scene()
    sky = next(x for x in scene.findall("asset/texture") if x.get("type") == "skybox")
    _, ground_material, ground = _floor_style(scene)
    spec.add_texture(name="native_walking_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=_numbers(sky.get("rgb1")), rgb2=_numbers(sky.get("rgb2")),
                     width=int(sky.get("width")), height=int(sky.get("height")))
    spec.add_texture(name="native_walking_ground", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=_numbers(ground.get("rgb1")), rgb2=_numbers(ground.get("rgb2")),
                     mark={"cross":mujoco.mjtMark.mjMARK_CROSS,"edge":mujoco.mjtMark.mjMARK_EDGE}.get(ground.get("mark"),mujoco.mjtMark.mjMARK_NONE),
                     markrgb=_numbers(ground.get("markrgb","0 0 0")),
                     width=int(ground.get("width")), height=int(ground.get("height")))
    material = spec.add_material(name="native_walking_ground", texuniform=True,
                                 texrepeat=_numbers(ground_material.get("texrepeat","1 1")),
                                 reflectance=float(ground_material.get("reflectance","0")))
    material.textures[1] = "native_walking_ground"
    return "native_walking_ground"


def _reference_texture(element, width, height):
    import mujoco
    texture = deepcopy(element)
    texture.attrib.update(name="native_reference", width=str(width), height=str(height))
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    asset.append(texture)
    if texture.get("type") == "2d":
        ET.SubElement(asset, "material", name="native_reference", texture="native_reference")
        ET.SubElement(ET.SubElement(root, "worldbody"), "geom", type="plane", size="0 0 .1", material="native_reference")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model.tex_data.copy()


def apply_native_style(model):
    """Retint only visual textures/lights and the world's floor material.

    This intentionally leaves geometry shapes, positions, groups, alpha,
    collision masks, mass/inertia, controls and depth-camera settings intact.
    """
    import mujoco
    scene = _scene()
    sky = next(x for x in scene.findall("asset/texture") if x.get("type") == "skybox")
    walking_ground,walking_material,floor = _floor_style(scene)
    floor_materials = set()
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            material = int(model.geom_matid[i])
            if material >= 0:
                floor_materials.add(material)
                model.geom_rgba[i,:3] = _numbers(walking_ground.get("rgba",".5 .5 .5 1"))[:3]
    floor_textures = {int(texture) for material in floor_materials
                      for texture in model.mat_texid[material] if texture >= 0}
    for i in range(model.ntex):
        is_sky = model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX
        if not is_sky and i not in floor_textures:
            continue
        pixels = _reference_texture(sky if is_sky else floor, int(model.tex_width[i]), int(model.tex_height[i]))
        offset = int(model.tex_adr[i])
        size = int(model.tex_width[i] * model.tex_height[i] * model.tex_nchannel[i])
        if pixels.size != size:
            raise RuntimeError("Walking visual texture channel mismatch")
        model.tex_data[offset:offset+size] = pixels
    for material in floor_materials:
        model.mat_texrepeat[material] = _numbers(walking_material.get("texrepeat","1 1"))
        model.mat_texuniform[material] = True
        model.mat_reflectance[material] = float(walking_material.get("reflectance","0"))
        model.mat_rgba[material, :3] = _numbers(walking_material.get("rgba","1 1 1 1"))[:3]
    light = scene.find("visual/headlight")
    for key in ("ambient", "diffuse", "specular"):
        getattr(model.vis.headlight, key)[:] = _numbers(light.get(key))
    model.vis.rgba.haze[:] = _numbers(scene.find("visual/rgba").get("haze"))
    model.vis.global_.azimuth = 150.
    model.vis.global_.elevation = -20.
    # Extent is a camera/visual statistic. Keep a robot-sized orbit/zoom basis
    # instead of scaling the near plane by the whole twelve-metre course.
    model.stat.extent = .8
    model.vis.map.znear = .05
    model.vis.map.zfar = 80.
    model.vis.map.fogstart = 10.
    model.vis.map.fogend = 50.
    model.vis.quality.shadowsize = 4096
    model.vis.quality.offsamples = 4
    if model.nlight:
        model.light_pos[0] = [0., 0., 1.5]
        model.light_dir[0] = [0., 0., -1.]
        model.light_type[0] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        model.light_diffuse[0] = [.7, .7, .7]
        model.light_ambient[0] = 0.
        model.light_specular[0] = 0.
        if model.nlight > 1:
            model.light_diffuse[1:] = 0.
            model.light_ambient[1:] = 0.
            model.light_specular[1:] = 0.
