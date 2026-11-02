# GUI 第三方库、源码与替换方法

本版使用 PySide6 / Shiboken6 / Qt **6.11.3**（Linux x86_64，CPython 3.10 ABI）。应用通过动态库和外部 Python 包入口加载这些库，采用 PyInstaller `onedir` 目录打包。

## 交付布局

```text
loong_gui_sim/
  loong_gui_sim                       # 应用入口
  _internal/
    PySide6/__init__.py               # 可替换的绑定入口
    PySide6/QtCore.abi3.so            # PySide6 扩展；同目录还有其他绑定
    PySide6/libpyside6.abi3.so.6.11
    PySide6/Qt/lib/libQt6*.so.6       # Qt 动态库
    PySide6/Qt/plugins/               # 与 Qt 配套的插件
    shiboken6/__init__.py
    shiboken6/Shiboken.abi3.so
    shiboken6/libshiboken6.abi3.so.6.11
  native/                            # 仿真原生辅助库
  runtime/                           # SDK 与模型配置
```

`_internal/` 顶层还有指向上述目录的相对符号链接，替换时保留它们。不应把旧 PyQt5 单文件包与新 `_internal/` 混用。

## 替换步骤

1. 退出 GUI 和它启动的仿真进程，复制整个发布目录作为测试副本。
2. 获取 `third_party/sources/` 中的对应源码，按各归档内的上游构建说明构建需要修改的库。也可以准备同平台、同 ABI 的官方运行库。应用没有对库进行本地修改；源码归档包含上游构建文件。`SOURCES.json` 记录精确版本、官方下载地址和哈希。
3. 在测试副本内替换相应文件。Qt、PySide6 和 Shiboken6 需要相互兼容；建议从 6.11.3 配套组件开始。替换 Qt 主库时也保留相容插件、ICU 依赖和 SONAME。不能用任意版本同名文件覆盖后就假设兼容。
4. 在测试副本根目录验证：

```bash
./start_sim.sh --qt-info
QT_QPA_PLATFORM=offscreen ./start_sim.sh --self-test
./start_sim.sh
```

`--qt-info` 输出实际导入路径及版本；自检不启动机器人控制链。正常启动后检查窗口、曲线和回放。GUI 启动不依赖文件哈希白名单，不要求重新编译应用来加载兼容库。

修改库之后，`check_release.py` 报出原始哈希不一致是预期的完整性提示，不是禁止替换，也不是启动门禁。请保留原始版本用于对照，按第三方许可证保留修改说明。

## 替换验证的范围

本次在单独的测试副本中重新放入官方 wheel 的 QtCore、Shiboken 动态库，并修改外部 PySide6 Python 入口加入探针。主程序成功读到修改标记，随后曲线与原生回放自检通过，见 `compliance/library-replacement.json`。这验证了外部库加载与修改机制，不等于所有跨版本库组合都相容，也不代表已重新编译并测试全部 Qt 模块。

## 许可证与权利

许可证及版权材料见 `licenses/gui/`、根目录 `NOTICE`、`LICENSES.md`。GNU LGPL v3 引用 GNU GPL v3，两者正文均需保留。允许按 LGPL 修改相关库，并允许为调试这些修改进行必要的反向工程。自有 GUI 应用部分按 `loong_gui_sim/LICENSE` 中的 MIT 条款分发。

请与二进制一起提供许可、声明、源码归档及本说明；不要只转发主程序或 Git LFS 指针。

参考：[Qt 的 LGPL 义务说明](https://www.qt.io/development/open-source-lgpl-obligations)、[Qt for Python 许可说明](https://doc.qt.io/qtforpython-6/licenses.html)。
