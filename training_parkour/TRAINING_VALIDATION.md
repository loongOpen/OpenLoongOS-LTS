# 越障训练整合复核

日期：2026-11-12。本次在独立 Python 3.12.14 环境、RTX 4090、驱动 580.65.06 上，重新完成了从头训练、保存、续训、权重审计和全部测试。原包 `environment-tested.json` 中的 CUDA 13 / 32 环境 12 轮记录属于原包已有结果，本次使用默认安装脚本锁定的 CUDA 12.8 组合。

机器可读结果：[training-validation-20261112.json](training-validation-20261112.json)。

## 环境与依赖

- Isaac Sim 6.0.0.1，包含 RL、asset、robot-setup；缺少后两项会导致 URDF importer 无法导入，现已纳入锁和预检。
- Isaac Lab 提交 `28a37cecdd433c22d9eabd6a5954add9f13a8951` / 元数据 6.1.11，显式安装 13 个 core 模块。
- PyTorch 2.11.0+cu128，RSL-RL 5.4.1，Warp 1.13.0，tensordict 0.12.3。
- 两行上游元数据补丁通过正向 `git apply --check` 和重复应用识别；不改仿真、任务、奖励或物理逻辑。
- 完整联合依赖解析和 `uv pip check`：**247 个包全部兼容**。锁中有 233 个第三方依赖，其余为 13 个上游 editable 模块和本项目。
- 实际 CUDA 张量分配、72 段动作、7569 帧、26 关节、27 个网格和 101 个必要资源全部通过预检。

## 实际训练与续训

训练从本项目原有的长中文路径启动。脚本使用短的 `/tmp/loong-lite.*` 临时目录，修复了 NVRTC 对长 UTF-8 临时路径报 `CUDA kernel build failed, code 6` 的问题；退出后清理本次临时目录。

| 检查 | 结果 |
|---|---|
| 从头训练 | 2 环境 × 24 步 × 2 轮，96 个环境转换，训练阶段约 4.94 秒 |
| 保存 | `model_0.pt`、`model_1.pt`；最终模型约 37.35 MB，含 actor/critic、PPO 优化器、判别器、判别器归一化和优化器 |
| 续训 | 从 `/tmp` 经仓库根 `train.sh parkour` 读取刚保存的 `model_1.pt`，追加 1 轮、48 个环境转换，约 2.6 秒 |
| 有限值 | 两轮训练和续训后的所有 checkpoint 张量均为有限值 |
| 完成证据 | 两次独立 nonce 均写入最终 checkpoint 的 `infos`，报告的实际文件大小、mtime 和 SHA-256 已独立核对 |

实际参数的最大绝对更新量：

| 参数组 | 两轮中第 0 → 第 1 轮 | 加载后续训 1 轮 |
|---|---:|---:|
| actor | 0.00144128 | 0.00424461 |
| critic | 0.00152506 | 0.00517091 |
| AMP 判别器 weight/bias | 0.0000201035 | 0.0000202563 |

RSL-RL 5.4.1 的 `iter` 是最后一个零基迭代索引；从 `iter=1` 追加一轮时，最终索引仍可为 1。因此完成校验使用 `final = start + requested - 1`，同时要求本次新的 checkpoint、nonce 和哈希，不以索引简单递增作为续训成功证据。

首次 Kit 注册表访问在本机遇到 DNS 返回代理地址而停滞，本次仅对测试进程把官方主机解析到通过 DoH 查询的 A 记录，保留原 HTTPS 主机名、SNI 和证书校验；未修改系统 DNS，也未将固定 IP 或该临时诊断方法写入发布安装脚本。正常安装与首次运行需要访问官方服务。基础 Kit 来自 Sim kernel，训练所需扩展按需取得，没有安装完整的 6 GB Kit 扩展缓存包。

## 回归与资源

- 完整 `pytest -q tests`：**42 passed**，63 个上游弃用警告。
- 其中 10 项预检和启动门槛测试，覆盖缺失资源、LFS 指针、URDF 改动，以及子进程返回 0 但无报告、旧 nonce、缺失/旧 checkpoint、错误进度和有效报告。
- 独立审查额外模拟了空文件、内容变更、错误哈希、坏 JSON 和子进程退出码 7，均被拒绝；真实两轮和单轮续训报告均通过独立校验。
- `validate_motion_data.py`：72 段动作的关节限位、四元数和关键体 FK 通过。
- `build_lite_mujoco.py --check`：64 组姿态的 FK、质量和惯量通过，最大位置误差 `4.44e-16 m`；生成 XML 与原发布哈希一致。
- wheel 构建成功，实查包含 72 个 PKL、27 个 STL、URDF、XML 和动作 manifest。
- `list_envs.py` 只列出两个 Lite 任务；训练不依赖其它机器人的工作区。

短训练证明了实际 PPO/AMP 更新、完整保存和恢复链路，未将随机起步策略描述为已经学会越障或收敛。
