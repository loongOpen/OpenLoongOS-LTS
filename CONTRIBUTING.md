# 参与贡献

一般问题见 [缺陷报告指南](docs/reporting-bugs.md)。涉及第三方组件时，先阅读 [组件许可](LICENSES.md)。

## 提交变更

1. 从最新 `main` 创建功能分支，或从 Fork 提交分支。
2. 保留修改范围内的原始版权、许可证和 NOTICE。不要把不同组件统一重新授权。
3. 不提交凭据、虚拟环境、训练日志和缓存；模型与二进制继续使用 Git LFS。
4. 新增或修改发布文件后更新 `FILE_HASHES.json`，清单不记录自身；不要重算未经核对的无关文件。
5. 运行 `python3 scripts/check_repository.py`。完整下载 LFS 后再运行 `python3 -B check_release.py`。
6. 提交 PR，写明变更、实际测试和未验证范围，由其他维护者审查后合入。

不要直接推送或强制推送 `main`。这是一项贡献规则；技术强制还需要平台管理员启用
[仓库维护设置](docs/repository-maintenance.md) 中的分支保护。

仓库自动检查验证文件清单、许可证材料、LFS 指针和启动脚本语法，不替代 GUI 图形实测、CUDA 训练或许可证人工审查。
