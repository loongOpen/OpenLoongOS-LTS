"""Standalone Lite simulation utilities: depth, joint PD, terrain and viewer."""
from __future__ import annotations
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shlex
import sys
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

CAMERA_WIDTH, CAMERA_HEIGHT = 64, 36
CAMERA_HFOV, CAMERA_VFOV = 89.51, 58.29
DEPTH_FAR = 2.5
DEPTH_SHAPE = (18, 32)
DEPTH_LAGS = np.arange(35, -1, -5)
DEPTH_GEOM_GROUP = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)


def numbers(values) -> str:
    return ' '.join((f'{float(x):.12g}' for x in values))

def rpy_quaternion(rpy) -> np.ndarray:
    """URDF extrinsic XYZ roll/pitch/yaw -> MuJoCo WXYZ quaternion."""
    r, p, y = np.asarray(rpy) / 2
    cr, cp, cy = (math.cos(r), math.cos(p), math.cos(y))
    sr, sp, sy = (math.sin(r), math.sin(p), math.sin(y))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])

def yaw_from_matrix(rotation: np.ndarray) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])

def yaw_free_tan_norm(rotation: np.ndarray) -> np.ndarray:
    yaw = yaw_from_matrix(rotation)
    c, s = (math.cos(yaw), math.sin(yaw))
    local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ rotation
    return np.concatenate((local[:, 0], local[:, 2]))

def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        candidates = [candidate for candidate in path.glob('model_*.pt') if candidate.is_file() and candidate.stem.removeprefix('model_').isdecimal()]
        if not candidates:
            raise FileNotFoundError(f'No model_<iteration>.pt checkpoint in {path}; pass a PT file explicitly.')
        path = max(candidates, key=lambda candidate: int(candidate.stem.removeprefix('model_')))
    if not path.is_file() or path.suffix.lower() != '.pt':
        raise ValueError(f'Expected a PT checkpoint or a training directory, got: {path}')
    return path

def add_stairs(world: ET.Element, step_height: float) -> None:
    """Straight up/platform/down test course; first riser 2 m in front of spawn."""
    width, count, plateau = (0.35, 6, 3.0)
    heights = [step_height * i for i in range(1, count + 1)]
    lengths = [width] * (count - 1) + [plateau] + [width] * (count - 1)
    heights += heights[-2::-1]
    x = 2.0
    for index, (height, length) in enumerate(zip(heights, lengths)):
        ET.SubElement(world, 'geom', name=f'stairs_{index}', type='box', group='0', pos=numbers([x + length / 2, 0, height / 2]), size=numbers([length / 2, 2, height / 2]), friction='1 0.005 0.0001', condim='3', solref='0.01 1', rgba='0.5 0.55 0.6 1')
        x += length

def add_box_terrain(root: ET.Element) -> np.ndarray:
    """Add rough ground and three fixed boxes; return normalized height samples."""
    start, end, half_width, resolution = (2.0, 12.0, 2.0, 0.1)
    ncol = round((end - start) / resolution) + 1
    nrow = round(2 * half_width / resolution) + 1
    heights = np.random.default_rng(42).uniform(size=(nrow, ncol))
    for _ in range(3):
        padded = np.pad(heights, 1, mode='reflect')
        heights = sum((padded[i:i + nrow, j:j + ncol] for i in range(3) for j in range(3))) / 9
    heights -= heights.min()
    heights /= heights.max()
    x, y = np.meshgrid(np.linspace(start, end, ncol), np.linspace(-half_width, half_width, nrow))
    edge_distance = np.minimum(np.minimum(x - start, end - x), half_width - np.abs(y))
    heights *= np.clip(edge_distance / 0.5, 0, 1)
    heights /= heights.max()
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'hfield', name='box_rough_hfield', nrow=str(nrow), ncol=str(ncol), size=numbers([(end - start) / 2, half_width, 0.1, 0.1]))
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='box_rough_ground', type='hfield', hfield='box_rough_hfield', pos=numbers([(start + end) / 2, 0, 0]), group='0', contype='1', conaffinity='1', friction='1 0.005 0.0001', condim='3', solref='0.01 1', rgba='0.50 0.62 0.52 1')
    for center, height in ((4.5, 0.2), (8.0, 0.3), (11.5, 0.4)):
        ET.SubElement(world, 'geom', name=f'box_obstacle_{round(height * 100)}cm', type='box', group='0', pos=numbers([center, 0, height / 2]), size=numbers([0.4, 0.75, height / 2]), contype='1', conaffinity='1', friction='1 0.005 0.0001', condim='3', solref='0.01 1', rgba='0.85 0.55 0.20 1')
    return heights

def configure_scene_lighting(root: ET.Element) -> None:
    """Use soft daylight in the viewer and RGB captures of the runtime model."""
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    headlight = visual.find('headlight')
    if headlight is None:
        headlight = ET.SubElement(visual, 'headlight')
    headlight.set('active', '1')
    headlight.set('ambient', '0.32 0.32 0.32')
    headlight.set('diffuse', '0.55 0.55 0.55')
    headlight.set('specular', '0.1 0.1 0.1')
    for index, light in enumerate(root.findall('worldbody/light')):
        light.set('ambient', '0.03 0.03 0.03')
        light.set('diffuse', '0.6 0.6 0.6' if index == 0 else '0.35 0.35 0.35')
        light.set('specular', '0.1 0.1 0.1')
        light.set('castshadow', 'true' if index == 0 else 'false')
    for texture in root.findall('asset/texture'):
        if texture.get('type') == 'skybox' and texture.get('builtin') == 'gradient':
            texture.set('rgb1', '0.6 0.75 0.9')
            texture.set('rgb2', '0.85 0.9 0.96')
        elif texture.get('name') == 'texplane' and texture.get('builtin') == 'checker':
            texture.set('rgb1', '0.55 0.61 0.67')
            texture.set('rgb2', '0.4 0.46 0.52')
            texture.set('markrgb', '0.85 0.88 0.92')

class HistoryBuffer:

    def __init__(self, length: int, shape: tuple[int, ...]):
        self.buffer = np.empty((length, *shape), dtype=np.float32)
        self.initialized = False

    def push(self, value: np.ndarray) -> None:
        if not self.initialized:
            self.buffer[:] = value
            self.initialized = True
        else:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = value

def build_local_rays() -> np.ndarray:
    aperture_h = 2 * math.tan(math.radians(CAMERA_HFOV) / 2)
    aperture_v = 2 * math.tan(math.radians(CAMERA_VFOV) / 2)
    y = (0.5 - (np.arange(CAMERA_WIDTH) + 0.5) / CAMERA_WIDTH) * aperture_h
    z = (0.5 - (np.arange(CAMERA_HEIGHT) + 0.5) / CAMERA_HEIGHT) * aperture_v
    yy, zz = np.meshgrid(y, z)
    rays = np.stack((np.ones_like(yy), yy, zz), axis=-1).reshape(-1, 3)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)

def postprocess_depth(depth: np.ndarray) -> np.ndarray:
    image = depth.reshape(CAMERA_HEIGHT, CAMERA_WIDTH)[18:, 16:-16].astype(np.float32)
    kernel = np.exp(-np.arange(-1, 2, dtype=np.float32) ** 2 / 2)
    kernel /= kernel.sum()
    padded = np.pad(image, 1, mode='reflect')
    blurred = np.zeros(DEPTH_SHAPE, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            blurred += kernel[i] * kernel[j] * padded[i:i + 18, j:j + 32]
    np.nan_to_num(blurred, copy=False, nan=DEPTH_FAR, posinf=DEPTH_FAR, neginf=0)
    return np.clip(blurred, 0, DEPTH_FAR) / DEPTH_FAR

class DepthCamera:

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        position, rotation = self.pose(data)
        rays = np.ascontiguousarray(self.local_rays @ rotation.T).ravel()
        arguments = (model, data, position, rays, self.geom_group, True, -1, self.geomid, self.distance)
        if self.with_normals:
            try:
                mujoco.mj_multiRay(*arguments, None, len(self.local_rays), 1000000.0)
            except TypeError:
                self.with_normals = False
                mujoco.mj_multiRay(*arguments, len(self.local_rays), 1000000.0)
        else:
            mujoco.mj_multiRay(*arguments, len(self.local_rays), 1000000.0)
        depth = self.distance * self.local_rays[:, 0]
        depth[(self.geomid < 0) | ~np.isfinite(depth) | (self.distance < 0)] = DEPTH_FAR
        frame = postprocess_depth(np.clip(depth, 0, DEPTH_FAR))
        self.history.push(frame)
        return self.history.buffer[-1 - DEPTH_LAGS][None].copy()

class JointPositionSimulation:

    def _joint_id(self, name: str) -> int:
        for candidate in (name, name.removesuffix('_joint')):
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
            if index >= 0:
                return index
        raise ValueError(f'Missing MuJoCo joint: {name}')

    def observation(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.base_id, velocity, 1)
        if self.heading is not None:
            error = (self.heading - yaw_from_matrix(rotation) + math.pi) % (2 * math.pi) - math.pi
            self.command[2] = np.clip(error, -1, 1)
        terms = (velocity[:3], yaw_free_tan_norm(rotation), self.command, self.data.qpos[self.qpos_ids], self.data.qvel[self.qvel_ids], self.last_action)
        for history, value in zip(self.histories, terms):
            history.push(value)
        proprio = np.concatenate([h.buffer.ravel() for h in self.histories])[None]
        return (proprio, self.camera.capture(self.model, self.data))

    def control_step(self) -> None:
        proprio, depth = self.observation()
        self.last_action[:] = self.policy(proprio, depth)
        self.target = self.q0 + self.action_scale * self.last_action

    def physics_step(self) -> None:
        target = self.target
        q, dq = (self.data.qpos[self.qpos_ids], self.data.qvel[self.qvel_ids])
        torque = np.clip(self.kp * (target - q) - self.kd * dq, -self.effort, self.effort)
        torque[(dq >= self.speed) & (torque > 0) | (dq <= -self.speed) & (torque < 0)] = 0
        self.data.ctrl[self.motor_ids] = torque
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('MuJoCo state became non-finite.')

class KeyboardController:
    """The reference script's pygame KEYDOWN/KEYUP semantics, polled at physics rate."""

    def __init__(self, pygame_module):
        self.pygame = pygame_module
        self.r_pressed = False
        self._fonts = None
        self._window = None

    def pin_window(self) -> None:
        """Keep the pygame display above other windows without changing its surface."""
        import ctypes
        from pygame._sdl2.video import Window
        self._window = Window.from_display_module()
        sdl = ctypes.CDLL(self.pygame.base.__file__)
        sdl.SDL_GetWindowFromID.argtypes = [ctypes.c_uint32]
        sdl.SDL_GetWindowFromID.restype = ctypes.c_void_p
        sdl.SDL_SetWindowAlwaysOnTop.argtypes = [ctypes.c_void_p, ctypes.c_int]
        sdl.SDL_SetWindowAlwaysOnTop.restype = None
        window = sdl.SDL_GetWindowFromID(self._window.id)
        if not window:
            raise RuntimeError('Cannot access the Keyboard Window to enable always-on-top.')
        sdl.SDL_SetWindowAlwaysOnTop(window, 1)
        self._window.focus()

    def draw_command(self, simulation: JointPositionSimulation) -> None:
        """Draw commands after policy sampling, including heading-controller updates."""
        pg = self.pygame
        if self._fonts is None:
            pg.font.init()
            self._fonts = tuple((pg.font.Font(None, size) for size in (24, 32, 20, 18)))
        title_font, value_font, label_font, small_font = self._fonts
        screen = pg.display.get_surface()
        screen.fill((18, 23, 31))
        foreground, muted, separator = ((235, 240, 246), (158, 177, 197), (48, 61, 78))
        screen.blit(title_font.render('Velocity command', True, foreground), (16, 12))
        pg.draw.line(screen, separator, (16, 38), (304, 38))
        rows = (('vx / forward', 'm/s'), ('vy / lateral', 'm/s'), ('wz / turn', 'rad/s'))
        for y, (label, unit), value in zip((51, 93, 135), rows, simulation.command):
            screen.blit(label_font.render(label, True, muted), (16, y + 7))
            text = value_font.render(f'{float(value):+.2f}', True, foreground)
            screen.blit(text, (242 - text.get_width(), y))
            screen.blit(small_font.render(unit, True, muted), (254, y + 8))
        pg.draw.line(screen, separator, (16, 181), (304, 181))
        screen.blit(small_font.render('W/S: vx   A/D: vy   Q/E: wz', True, muted), (16, 191))
        screen.blit(small_font.render('R: max 1.50 m/s   T: reset', True, muted), (16, 213))
        pg.display.flip()

    def poll(self, simulation: JointPositionSimulation) -> bool:
        pg = self.pygame
        changed = False
        movement = {pg.K_w: (0, 0.6), pg.K_s: (0, -0.8), pg.K_a: (1, 0.5), pg.K_d: (1, -0.5), pg.K_q: (2, 1.0), pg.K_e: (2, -1.0)}
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN:
                if event.key in movement:
                    axis, value = movement[event.key]
                    simulation.command[axis] = value
                    simulation.heading = None
                    changed = True
                elif event.key == pg.K_r:
                    self.r_pressed = True
                    simulation.heading = None
                elif event.key == pg.K_t:
                    simulation.reset()
                    print('[RESET] T: robot, commands and histories reset.')
                    changed = True
            elif event.type == pg.KEYUP:
                if event.key in movement:
                    simulation.command[movement[event.key][0]] = 0.0
                    changed = True
                elif event.key == pg.K_r:
                    self.r_pressed = False
        if self.r_pressed:
            simulation.command[0] = min(simulation.command[0] + 0.01, 1.5)
            changed = True
        if changed:
            vx, vy, wz = simulation.command
            print(f'[COMMAND] vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}')
        return True

def draw_camera(viewer, simulation: JointPositionSimulation) -> None:
    position, rotation = simulation.camera.pose(simulation.data)
    rays = simulation.camera.local_rays[[0, 63, -1, -64]]
    corners = position + rays / rays[:, :1] * 0.2 @ rotation.T
    segments = [(position, point) for point in corners]
    segments += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    viewer._markers[:] = []
    for start, end in segments:
        viewer.add_marker(pos=start, fromto=np.concatenate((start, end)), size=np.array([4.0, 0, 0]), rgba=np.array([1, 0.2, 0.2, 0.9], dtype=np.float32), type=mujoco.mjtGeom.mjGEOM_LINE, label='')

def save_frame(path: Path, simulation: JointPositionSimulation) -> None:
    from PIL import Image
    mujoco.mj_forward(simulation.model, simulation.data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = simulation.data.xpos[simulation.base_id] + np.array([0, 0, 0.2])
    camera.distance, camera.azimuth, camera.elevation = (3.5, 135, -20)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(simulation.model, height=480, width=640) as renderer:
        renderer.update_scene(simulation.data, camera=camera, scene_option=option)
        pixels = renderer.render().copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    print(f'[IMAGE] {path.resolve()}')

def install_viewer_compat(viewer) -> None:
    """Reference viewer fixes for removed MjvGeom texture fields and integer skin selection."""
    import glfw

    def add_marker(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError('MuJoCo viewer marker capacity exceeded.')
        geom = self.scn.geoms[self.scn.ngeom]
        kind = marker.get('type', mujoco.mjtGeom.mjGEOM_BOX)
        size = np.asarray(marker.get('size', [0.1, 0.1, 0.1]), dtype=np.float64)
        rgba = np.asarray(marker.get('rgba', [1, 1, 1, 1]), dtype=np.float32)
        mujoco.mjv_initGeom(geom, kind, size, np.asarray(marker.get('pos', np.zeros(3))), np.asarray(marker.get('mat', np.eye(3))).ravel(), rgba)
        geom.dataid, geom.objid = (-1, -1)
        geom.objtype, geom.category = (mujoco.mjtObj.mjOBJ_UNKNOWN, mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = marker.get('label') or ''
        if 'fromto' in marker:
            endpoints = np.asarray(marker['fromto'], dtype=np.float64).reshape(2, 3)
            mujoco.mjv_connector(geom, kind, float(size[0]), endpoints[0], endpoints[1])
            geom.rgba[:] = rgba
        self.scn.ngeom += 1
    viewer._add_marker_to_scene = add_marker.__get__(viewer, type(viewer))

    def mouse_button(window, button, action, mods):
        viewer._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
        viewer._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS
        x, y = glfw.get_cursor_pos(window)
        viewer._last_mouse_x, viewer._last_mouse_y = (int(viewer._scale * x), int(viewer._scale * y))
        now = glfw.get_time()
        double_click = []
        for side, pressed, threshold in (('left', viewer._button_left_pressed, 0.3), ('right', viewer._button_right_pressed, 0.2)):
            previous = getattr(viewer, f'_last_{side}_click_time')
            double = pressed and previous is not None and (0.01 < now - previous < threshold)
            setattr(viewer, f'_{side}_double_click_pressed', double)
            if pressed:
                setattr(viewer, f'_last_{side}_click_time', now)
            double_click.append(double)
        control = bool(mods & glfw.MOD_CONTROL)
        perturb = 0
        if control and viewer.pert.select > 0:
            if viewer._button_right_pressed:
                perturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if viewer._button_left_pressed:
                perturb = mujoco.mjtPertBit.mjPERT_ROTATE
            if perturb and (not viewer.pert.active):
                mujoco.mjv_initPerturb(viewer.model, viewer.data, viewer.scn, viewer.pert)
        viewer.pert.active = perturb
        if any(double_click):
            width, height = glfw.get_window_size(window)
            point = np.zeros(3)
            geom, flex, skin = (np.zeros(1, dtype=np.int32) for _ in range(3))
            arguments = (viewer.model, viewer.data, viewer.vopt, width / height, x / width, 1 - y / height, viewer.scn, point, geom)
            if int(mujoco.__version__.split('.')[0]) >= 3:
                body = mujoco.mjv_select(*arguments, flex, skin)
            else:
                body = mujoco.mjv_select(*arguments, skin)
            if double_click[1]:
                if body >= 0:
                    viewer.cam.lookat[:] = point
                if control and body > 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid, viewer.cam.fixedcamid = (body, -1)
            elif body >= 0:
                viewer.pert.select, viewer.pert.skinselect = (body, int(skin[0]))
                viewer.pert.localpos[:] = viewer.data.xmat[body].reshape(3, 3).T @ (point - viewer.data.xpos[body])
            else:
                viewer.pert.select, viewer.pert.skinselect = (0, -1)
            viewer.pert.active = 0
        if action == glfw.RELEASE:
            viewer.pert.active = 0
    glfw.set_mouse_button_callback(viewer.window, mouse_button)

@contextmanager
def interactive_viewer(simulation: JointPositionSimulation):
    try:
        import pygame
        import mujoco_viewer
    except ModuleNotFoundError as error:
        if error.name not in {'pygame', 'mujoco_viewer'}:
            raise
        command = shlex.join([sys.executable, '-m', 'pip', 'install', 'pygame', 'mujoco-python-viewer'])
        raise SystemExit(f'GUI dependencies are missing. Install with:\n  {command}') from None
    os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
    os.environ.setdefault('SDL_FRAMEBUFFER_ACCELERATION', '0')
    viewer = None
    try:
        pygame.display.init()
        pygame.display.set_caption('Keyboard Window')
        pygame.display.set_mode((320, 240))
        keyboard = KeyboardController(pygame)
        viewer = mujoco_viewer.MujocoViewer(simulation.model, simulation.data)
        install_viewer_compat(viewer)
        viewer.vopt.geomgroup[3] = 0
        if simulation.args.follow_view:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid, viewer.cam.fixedcamid = (simulation.camera.body_id, -1)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (3.0, 0.0, -20.0)
        if simulation.args.no_realtime:
            import glfw
            glfw.swap_interval(0)
        keyboard.pin_window()
        keyboard.draw_command(simulation)
        yield (viewer, keyboard)
    finally:
        if viewer is not None and viewer.is_alive:
            viewer.close()
        pygame.quit()
