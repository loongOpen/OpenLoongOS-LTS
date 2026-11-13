# libxcb-cursor

Qt 的 XCB 平台插件使用此库。发布包仅额外包含该 cursor 动态库，由 PyInstaller 的 `_internal/` 库搜索路径直接加载；其他 XCB/OpenGL 系统依赖继续按快速入门安装。

文件从 SHA-256 校验通过的 Ubuntu 官方二进制包原样提取。完整来源、版本、包校验值及库校验值见 `SOURCE.json`，原始许可声明见 `copyright`。私有构建脚本 `bundle_xcb_cursor.py` 会在每次 GUI 构建后重新纳入此库。
