# Isaac Lab 与 Isaac Sim 6 依赖元数据兼容补丁

上游固定提交：[28a37cecdd433c22d9eabd6a5954add9f13a8951](https://github.com/isaac-sim/IsaacLab/commit/28a37cecdd433c22d9eabd6a5954add9f13a8951)。

`isaaclab-sim6-metadata.patch` 只修改两个安装声明：

| 文件 | 上游声明 | 本项目兼容声明 | 原因 |
|---|---|---|---|
| `source/isaaclab/setup.py` | `coverage==7.6.1` | `coverage==7.4.4` | 与 `isaacsim-kernel==6.0.0.1` 的固定版本一致；仅涉及测试覆盖工具 |
| `source/isaaclab_rl/setup.py` | `packaging<24` | `packaging==26.0` | 与 `isaacsim-core==6.0.0.1` 的固定版本一致；相关代码使用普通 `version.parse` / `Version` |

不修改上游仿真、任务、策略、奖励、物理参数或许可证文件。上游源码保留 BSD-3-Clause 原始许可头。
`scripts/prepare_isaaclab.py` 先核对 Git 提交，再执行 `git apply --check`；已应用时验证反向补丁可用，
其它提交或无法匹配的文件会退出，不覆盖未知源码。

安装脚本显式安装该提交的 core 模块，避免上游三个与本组合冲突的入口：
`isaaclab[isaacsim/all]` 固定 Sim 5.1、`isaaclab_rl[rsl-rl]` 固定 RSL-RL 5.0.1，
以及 `isaaclab.sh -i core` 自动重装 Torch 2.10。所有依赖在同一次解析中匹配，安装结束执行 `uv pip check`。

联合解析记录与实际训练范围见 [训练复核记录](../TRAINING_VALIDATION.md)。
