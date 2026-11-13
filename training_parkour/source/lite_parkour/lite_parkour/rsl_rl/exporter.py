import copy
import os
import torch


def ensure_onnx_ir_compatibility(model_path: str, max_ir_version: int = 9) -> tuple[int, int]:
    """Ensure an exported ONNX model does not exceed a runtime's maximum IR version.

    PyTorch's current ONNX exporter emits IR version 10 even when the requested
    operator-set version is compatible with older ONNX Runtime releases. ONNX
    Runtime 1.16, used by the Jetson deployment environment, supports IR up to
    version 9. Since the exported policies do not use IR-10-only features, the
    model header can be downgraded without touching its graph or external
    weights.

    The converted model is checked before atomically replacing the original.

    Args:
        model_path: Path to the exported ``.onnx`` model.
        max_ir_version: Maximum allowed ONNX IR version.

    Returns:
        A tuple containing ``(original_ir_version, final_ir_version)``.
    """
    if max_ir_version < 1:
        raise ValueError(f"max_ir_version must be positive, got {max_ir_version}")

    import onnx

    model = onnx.load(model_path, load_external_data=False)
    original_ir_version = int(model.ir_version)
    if original_ir_version <= max_ir_version:
        print(
            f"[INFO] ONNX IR compatibility already satisfied: "
            f"IR {original_ir_version} <= IR {max_ir_version}"
        )
        return original_ir_version, original_ir_version

    model.ir_version = max_ir_version
    temporary_path = f"{model_path}.ir{max_ir_version}.tmp.onnx"
    try:
        onnx.save_model(model, temporary_path)
        # Passing a path lets the checker resolve external tensor data relative
        # to the model directory without loading it into this process first.
        onnx.checker.check_model(temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print(
        f"[INFO] Converted exported ONNX model from IR {original_ir_version} "
        f"to IR {max_ir_version}: {model_path}"
    )
    return original_ir_version, max_ir_version


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            input_size = getattr(self.actor, "input_dim", None)
            if input_size is None:
                input_size = self.actor[0].in_features
            obs = torch.zeros(1, input_size)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
