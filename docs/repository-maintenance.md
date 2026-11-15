# 仓库维护与发布设置

以下设置需要在 GitHub 和 AtomGit 各自配置。提交本文件不会自动修改平台设置。

## main 分支保护

- 对 `main` 启用保护规则，禁止直接推送、强制推送和删除。
- 仅允许通过 PR 合入，至少 1 名非作者审核通过；新提交使旧批准失效。
- 处理完审查对话后再合并；管理员也遵守规则，不配置日常绕过权限。
- 首次成功运行 CI 后，将 `repository-check` 设置为必须通过的检查。
- AtomGit 先启用流水线并配置可用 Runner，再设置合并必须通过流水线；不要把尚不存在的检查设为必需。

GitHub 入口为 Settings → Rules / Branches；没有设置权限时由仓库管理员操作。
AtomGit 在仓库设置或分支列表中的保护分支入口配置，按当前界面选择对应权限项。
完成后用普通贡献账号验证：直接推送 main 被拒绝，审核和检查未通过的 PR 无法合并。
不要用强推正式主分支来测试规则。

## Issue 标签

在两个平台分别建立以下标签；模板字段只引用标签，不会代替平台创建。

| 标签 | 颜色 | 用途 |
| --- | --- | --- |
| bug | d73a4a | 可复现的缺陷 |
| needs-triage | fbca04 | 等待确认与分流 |
| security | b60205 | 已适合公开的安全事项；标签不改变可见性 |

GitHub 模板位于 `.github/ISSUE_TEMPLATE/`，AtomGit 当前界面使用 `.gitcode/ISSUE_TEMPLATE/`，
并提供 `.atomgit/ISSUE_TEMPLATE/` Markdown 模板兼容原入口。默认分流账号为 BAO123 / bjwwww。

## 自动检查

GitHub 使用 `.github/workflows/repository-check.yml`；AtomGit 使用 `.gitcode/workflows/repository-check.yml`。
检查以只读权限运行，不使用项目密钥，不启动机器人进程，也不安装 GPU 训练环境。
`python3 scripts/check_repository.py` 支持完整文件和 Git LFS 指针：普通文件核对内容哈希，指针核对登记的对象 SHA256。
指针通过不代表对象已上传；发版前必须在新目录完成 `git lfs pull` 并运行完整检查。

## 正式发布

1. 从已审核的 main 提交创建版本标签，记录提交 SHA 和验证范围。
2. 完整获取 LFS，执行 `python3 -B check_release.py`；需要运行验证时按 `VALIDATION.md` 的环境进行。
3. 仅按 `FILE_HASHES.json` 与清单文件本身打包，不带 `.git`、虚拟环境和运行日志。
4. 包内保留 LICENSE、LICENSES.md、NOTICE、licenses/、对应第三方源码以及各子项目许可材料。
5. 在两个平台创建 Release，上传完整压缩包及 `.sha256`，说明三套训练源码和统一 GUI 的范围。
6. 从平台下载附件后核对 SHA256、解压并检查。平台自动生成的源码 ZIP 不保证包含 LFS 实体。

Git 推送代码、创建标签和发布带附件的 Release 是不同操作，不能相互代替。
根 LICENSE 保持组件索引；不得为消除平台的 NOASSERTION 提示，把全部组件标为 Apache-2.0。

## 参考

- [AtomGit Issue 模板](https://docs.atomgit.com/docs/help/home/org_project/issue/issue_template/)
- [AtomGit 原版模板目录](https://docs.openatom.tech/repo/issue_tempalte/)
- [AtomGit 流水线触发配置](https://docs.atomgit.com/docs/help/home/org_project/pipeline/writing-pipelines/configure-triggers/)
- [GitHub 分支保护](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
