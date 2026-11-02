# 组件许可与来源

本项目由分别授权的组件组成。根目录 `LICENSE` 是许可索引，不把所有组件重新授权为同一种许可证。

| 内容 | 来源与适用许可 |
|---|---|
| `mjlab_amp_ovision/` | 在上游 mjlab 基础上二次开发；保留 Apache-2.0、源码版权声明及 rsl_rl 等依赖的各自许可 |
| `loong_gui_sim/runtime/license` | 自研 SDK 原有 MIT 许可文本及原作者声明，按其原有范围适用 |
| `loong_gui_sim/licenses/` | 自研与上游材料混合；保留 SDK、driver、关节映射、MuJoCo、GLFW、Quill 等各自 LICENSE / NOTICE |
| GUI 自有应用部分 | 本次迁移不改变其权属，不自动套用训练框架或 SDK 许可证；分发许可待权利人明确，见复核记录 |
| `models/` | 按维护者说明归类为自研；本次修改不变更模型及素材原有授权范围 |

## 本候选版 GUI 的实际第三方组件

| 组件 | 版本 / 范围 | 许可材料 |
|---|---|---|
| PySide6 / Shiboken6 | 6.11.3；运行绑定与支持代码 | LGPL-3.0，`licenses/gui/pyside-setup/`；对应源码 `third_party/sources/` |
| Qt | 6.11.3；Core、Gui、Widgets、OpenGL、Network、DBus、Test、XcbQpa、Svg 和清单中所列插件 | 所用 Qt 运行库采用 LGPL-3.0；保留 qtbase、qtsvg、qtimageformats 各自许可证及内含第三方声明 |
| ICU | 73；wheel 自带 `libicu*.so.73` | Unicode / ICU 及所含数据许可，`licenses/gui/ICU/LICENSE` |
| CPython | 3.10.12，Ubuntu 22.04 构建 | PSF 与历史、内含组件许可，`licenses/gui/CPython/`，不能缩减成单独一份 PSF 文本 |
| PyInstaller bootloader | 6.22.3 | GPL-2.0-or-later **附 Bootloader Exception**，`licenses/gui/PyInstaller/COPYING.txt`；例外不替代其他依赖义务 |
| Cython | 3.3.0，应用模块的构建工具 | `licenses/gui/Cython/`；上游声明生成输出不因此受 Cython 许可证约束 |
| ONNX Runtime（SDK 使用） | 1.21.0 | `licenses/onnxruntime/LICENSE` 与完整 `ThirdPartyNotices.txt` |

文件与归档模块清单见 [gui-inventory.json](compliance/gui-inventory.json)。本候选版未打入 PyQt5 / Qt5、Paramiko、Readline 模块及原控制台专用 Python 依赖。操作系统提供的动态库需按快速入门安装，不属于重新打包的 GUI 内置库。

PySide6 / Shiboken6 的 Python 支持入口、绑定 `.so`、Qt 动态库和插件均放在可替换目录中，不将其 Python 入口封入主程序。见 [库替换说明](docs/gui-library-replacement.md)、[NOTICE](NOTICE) 及 [库源码](third_party/sources/README.md)。允许按 LGPL 修改相关库，以及为调试这些修改而进行必要的反向工程。

`qt-source-attributions.json` 和 `upstream-notices/` 保留上游源码中的声明，涵盖范围可能大于本次打包的运行组件；不能把源码归档中每个工具、示例都当作应用链接依赖。GPL 许可文本也是 LGPL v3 所引用的必备文本，不因扫描命中而删除。

## 与历史版本的区别

历史目录 `licenses/loong_gui_ctrl/` 曾混放仿真 GUI 与独立机器人控制台的依赖材料。目录命名不表示第一阶段包含独立机器人控制台。新包按实际组件改用 `licenses/gui/`；旧二进制的实测清单保留为 `compliance/previous-gui-inventory.json`。

此次迁移只处理新候选包，不追溯消除旧 PyQt5 二进制分发的许可问题，也不改写远端 Git 历史。完整复核结论及剩余事项见 [复核记录](compliance/review-20261102.md)。
