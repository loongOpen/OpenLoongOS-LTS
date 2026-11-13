"""Batched URDF forward kinematics for Lite resource validation."""
import numpy as np
from scipy.spatial.transform import Rotation

def forward_kinematics(urdf, root, angles, count):
    """Batched link transforms relative to ``root``; ``angles`` maps joint name -> (count,)."""
    identity = np.broadcast_to(np.eye(4), (count, 4, 4))
    transforms = {root: identity.copy()}
    pending = list(urdf.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            local = identity.copy()
            origin = joint.find('origin')
            if origin is not None:
                local[:, :3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                local[:, :3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            if joint.get('type') != 'fixed':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                motion = identity.copy()
                motion[:, :3, :3] = Rotation.from_rotvec(angles.get(joint.get('name'), np.zeros(count))[:, None] * axis / np.linalg.norm(axis)).as_matrix()
                local = local @ motion
            transforms[joint.find('child').get('link')] = transforms[parent] @ local
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError(f'URDF is not a connected tree rooted at {root}')
    return transforms
