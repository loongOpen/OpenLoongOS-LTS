# 缺陷报告指南

[返回首页](../README.md) · [快速入门](quickstart.md) · [验证范围](../VALIDATION.md)

## 先做基本检查

1. 确认操作系统和图形环境满足快速入门要求，三套训练使用各自独立环境。
2. 在项目根目录运行 `python3 -B check_release.py`。Git 下载提示 LFS 指针时先执行 `git lfs pull`。
3. GUI 问题记录首次报错和 4 / 28 / 43 功能键值；训练问题先用对应 README 的小规模训练复现。
4. 搜索已有 Issue，避免重复提交。

## 在哪个平台反馈

- [GitHub Issue](https://github.com/loongOpen/OpenLoongOS-LTS/issues/new/choose)
- [AtomGit Issue](https://atomgit.com/openloong/OpenLoongOS-LTS/issues)

任选一个平台提交即可；如两边都有相关记录，请互相提供链接。
选择「缺陷报告」模板，填写版本、环境、复现步骤、预期结果、实际结果和已脱敏报错。
模板未显示时，可按上述字段直接填写。

## 保护日志中的信息

**请勿粘贴 Token、密钥、密码、证书私钥、私人连接信息或个人信息。**
检查日志、截图、命令行及配置文件后再上传；用 `[REDACTED]` 替换敏感内容，并保留定位问题所需的字段名和报错。
不要直接上传 `.env`、SSH 配置、完整环境变量、原始日志目录或 core dump。
如凭据已公开，请立即撤销或轮换；删除公开内容不能撤销已经泄露的凭据。

普通 Issue 是公开的。发现安全漏洞时不要提交可复现攻击细节、真实凭据或敏感数据。
目前本项目未公布专用安全邮箱与响应时限，也不把公开 Issue 当作私密通道。
在双方确认适当的私密接收方式前，只公开请求建立安全联络所必需的非敏感信息。

## 分类与处理

模板预设 `bug`、`needs-triage` 标签；平台标签未建立前可能不显示，维护者需按仓库维护指南创建。
`security` 只用于已经适合公开的安全事项；添加标签不会让 Issue 变为私密。
默认分流账号为 GitHub `BAO123` / AtomGit `bjwwww`，由维护团队转交对应模块负责人。
