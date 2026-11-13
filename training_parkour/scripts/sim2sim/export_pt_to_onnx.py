"""Export model_<iteration>.pt to exported/policy_<iteration>.onnx beside the PT file.

Dependencies: numpy, torch, onnx; optional output validation: onnxruntime.
The sim2sim script also uses these policy-loading and export functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEPTH_SHAPE = (18, 32)


def load_policy(checkpoint: Path):
    import torch
    from torch import nn

    torch.set_num_threads(1)  # a tiny network; more threads only add contention
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = state.get("actor_state_dict", state)
    if any(k.startswith("obs_normalizer") for k in actor):
        raise ValueError("checkpoint 带有 obs_normalizer，本脚本尚未实现归一化")
    mlp_keys = sorted({int(k.split(".")[1]) for k in actor if k.startswith("mlp.")})
    enc_keys = sorted({int(k.split(".")[2]) for k in actor if k.startswith("encoder_heads.depth.")})

    def stack(prefix, indices, act):
        layers = []
        for i, idx in enumerate(indices):
            w = actor[f"{prefix}{idx}.weight"]
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            if i < len(indices) - 1:
                layers.append(act())
        return nn.Sequential(*layers)

    conv_w = actor["cnns.depth.0.weight"]
    in_frames = conv_w.shape[1]

    class DepthActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(in_frames, conv_w.shape[0], conv_w.shape[2], padding=conv_w.shape[2] // 2),
                nn.ReLU(), nn.Flatten(),
            )
            self.encoder = stack("encoder_heads.depth.", enc_keys, nn.ReLU)
            self.mlp = stack("mlp.", mlp_keys, nn.ELU)

        def forward(self, obs, depth):
            return self.mlp(torch.cat([obs, self.encoder(self.cnn(depth))], dim=-1))

    net = DepthActor()
    remap = {}
    for k, v in actor.items():
        if k.startswith("cnns.depth."):
            remap["cnn." + k[len("cnns.depth."):]] = v
        elif k.startswith("encoder_heads.depth."):
            idx, attr = k[len("encoder_heads.depth."):].split(".")
            remap[f"encoder.{enc_keys.index(int(idx)) * 2}.{attr}"] = v
        elif k.startswith("mlp."):
            idx, attr = k[len("mlp."):].split(".")
            remap[f"mlp.{mlp_keys.index(int(idx)) * 2}.{attr}"] = v
    net.load_state_dict(remap, strict=True)
    net.eval()
    obs_dim = net.mlp[0].in_features - net.encoder[-1].out_features
    num_actions = net.mlp[-1].out_features
    return net, obs_dim, in_frames, num_actions


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(net, obs_dim, frames, out_dir: Path, meta: dict | None = None, *,
                depth_shape=DEPTH_SHAPE, filename="policy.onnx"):
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / filename
    dummy = (torch.zeros(1, obs_dim), torch.zeros(1, frames, *depth_shape))
    torch.onnx.export(net, dummy, str(onnx_path), input_names=["obs", "depth"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    if meta is not None:
        meta["onnx_sha256"] = sha256(onnx_path)
        (out_dir / "deploy.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"ONNX 已导出: {onnx_path}（未安装 onnxruntime，跳过校验）")
    else:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        o = rng.normal(size=(1, obs_dim)).astype(np.float32)
        d = rng.uniform(size=(1, frames, *depth_shape)).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(o), torch.from_numpy(d)).numpy()
        err = np.abs(sess.run(None, {"obs": o, "depth": d})[0] - ref).max()
        print(f"ONNX 已导出: {onnx_path}，与 torch 最大误差 {err:.2e}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="待转换的训练 checkpoint（.pt）")
    parser.add_argument("--output-dir", type=Path, help="导出目录，默认是 PT 文件所在目录下的 exported 文件夹")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint 文件不存在: {checkpoint}")
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else checkpoint.parent / "exported"
    filename = f"policy_{checkpoint.stem.removeprefix('model_')}.onnx"
    net, obs_dim, frames, num_actions = load_policy(checkpoint)
    print(f"PT: {checkpoint}")
    print(f"接口: obs[1,{obs_dim}] + depth[1,{frames},{DEPTH_SHAPE[0]},{DEPTH_SHAPE[1]}] -> actions[1,{num_actions}]")
    export_onnx(net, obs_dim, frames, out_dir, filename=filename)


if __name__ == "__main__":
    main()
