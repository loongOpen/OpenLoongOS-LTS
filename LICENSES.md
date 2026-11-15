# 组件许可与来源

本项目由分别授权的组件组成。根目录 `LICENSE` 是许可索引，不把所有组件重新授权为同一种许可证。

| 内容 | 来源与适用许可 |
|---|---|
| `training_walking/` | 在上游 mjlab 基础上二次开发；保留 Apache-2.0、源码版权声明及 rsl_rl 等依赖的各自许可 |
| `training_dance/` | mjlab 舞蹈训练源码；保留 `LICENSE`、`LICENSE-LINGLONG-SOURCE`、`NOTICE` 与源码中的上游声明 |
| `training_parkour/` | 项目级 Apache-2.0；具体文件按原始许可头适用，InstinctLab 派生组件含 CC BY-NC 4.0 非商业使用条款，详见 `training_parkour/NOTICE.md` |
| `loong_gui_sim/runtime/license` | 自研 SDK 原有 MIT 许可文本及原作者声明，按其原有范围适用 |
| `loong_gui_sim/licenses/` | 自研与上游材料混合；保留 SDK、driver、关节映射、MuJoCo、GLFW、Quill 等各自 LICENSE / NOTICE |
| GUI 自有应用部分 | 采用 [MIT](loong_gui_sim/LICENSE)，仅覆盖 GUI 自有应用代码；第三方运行库、SDK、原生辅助库及资源各自授权继续适用 |
| 新增跳舞／越障运行后端与原生 plan | `loong_gui_sim/runtime/python/` 及 `native/` 中带 MIT 标记的本项目代码；ONNX Runtime C API 头保留 `native/vendor/onnxruntime/LICENSE` |
| 跳舞原始附件 | `runtime/deploy/dance/` 和 `runtime/model/dance/` 保留用户提供的 ARM 库、ini、ONNX、refbin、NPZ；文件来源与原始哈希见 `ASSETS.json`，不因新 plan 的 MIT 标记而改变原文件授权 |
| `models/` | 自研模型；各资源按随附授权说明适用 |

## 非商业组件与许可证识别

`training_parkour/NOTICE.md` 和相关文件头标注的 InstinctLab 派生代码保留 **CC BY-NC 4.0**，
其非商业限制不因项目级 Apache-2.0、GUI 的 MIT、LFS 存储或打包方式而消失。
本仓库不能整体宣称所有组件都满足 OSI 开源定义，也不能宣称全部内容允许无条件商用。
若交付要求所有源码均符合 OSI 定义，需要取得相应权利人的兼容授权，或完成相关组件替换与训练、部署验证。
当前版本未完成这项替换，保留原功能和原许可。

根目录 LICENSE 是组件索引。平台显示 `NOASSERTION` 仅表示自动识别没有给出单一许可证结论，
不要为改变标签而覆盖第三方文件头或把全部目录重新标为 Apache-2.0。

参考：[CC 非商业条款说明](https://creativecommons.org/share-your-work/use-remix/cc-licenses/)、
[OSI 开源定义](https://opensource.org/osd)。

## 本版 GUI 的实际第三方组件

| 组件 | 版本 / 范围 | 许可材料 |
|---|---|---|
| PySide6 / Shiboken6 | 6.11.3；运行绑定与支持代码 | LGPL-3.0，`licenses/gui/pyside-setup/`；对应源码 `third_party/sources/` |
| Qt | 6.11.3；Core、Gui、Widgets、OpenGL、Network、DBus、Test、XcbQpa、Svg 和清单中所列插件 | 所用 Qt 运行库采用 LGPL-3.0；保留 qtbase、qtsvg、qtimageformats 各自许可证及内含第三方声明 |
| ICU | 73；wheel 自带 `libicu*.so.73` | Unicode / ICU 及所含数据许可，`licenses/gui/ICU/LICENSE` |
| CPython | 3.10.12，Ubuntu 22.04 构建 | PSF 与历史、内含组件许可，`licenses/gui/CPython/`，不能缩减成单独一份 PSF 文本 |
| PyInstaller bootloader | 6.22.3 | GPL-2.0-or-later **附 Bootloader Exception**，`licenses/gui/PyInstaller/COPYING.txt`；例外不替代其他依赖义务 |
| Cython | 3.3.0，应用模块的构建工具 | `licenses/gui/Cython/`；上游声明生成输出不因此受 Cython 许可证约束 |
| ONNX Runtime（SDK 使用） | 1.21.0 | `licenses/onnxruntime/LICENSE` 与完整 `ThirdPartyNotices.txt` |
| xcb-util-cursor | Ubuntu `libxcb-cursor0` 0.1.1-4ubuntu1，Linux x86_64 | MIT/X Consortium，原始版权文本和包／库来源及 SHA-256 见 `loong_gui_sim/licenses/libxcb-cursor/` |

运行文件清单见 [COMPONENTS.json](licenses/gui/COMPONENTS.json)。Qt XCB 所需的 cursor 库已放入 GUI `_internal/`；其余操作系统图形依赖按快速入门安装。

跳舞／越障的外部 Python 环境由 `loong_gui_sim/setup_runtime.sh` 按独立 `uv.lock` 安装 NumPy、MuJoCo、ONNX Runtime、SciPy、Pillow 及其依赖；这些依赖不封入 GUI 主程序，保留安装包自身的许可证。运行后端复用 `training_parkour/` 的模型与部署辅助代码，其原有许可及非商业条款继续适用，见该目录 `NOTICE.md`。

PySide6 / Shiboken6 的 Python 支持入口、绑定 `.so`、Qt 动态库和插件均放在可替换目录中，不将其 Python 入口封入主程序。见 [库替换说明](docs/gui-library-replacement.md)、[NOTICE](NOTICE) 及 [库源码](third_party/sources/README.md)。允许按 LGPL 修改相关库，以及为调试这些修改而进行必要的反向工程。

`qt-source-attributions.json` 和 `upstream-notices/` 保留上游源码中的声明，涵盖范围可能大于本次打包的运行组件；源码中的工具和示例不一定属于运行依赖。GPL 许可文本也是 LGPL v3 所引用的必备文本。
