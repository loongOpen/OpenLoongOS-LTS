# 组件许可与来源

本项目是多个组件的整合发布，不使用单一许可证覆盖所有内容。

- `mjlab_amp_ovision/`：保留 Apache-2.0 许可与源码版权声明，基于 mjlab，强化学习依赖 rsl_rl。
- `loong_gui_sim/runtime/license`：原 SDK 的许可文本，保留原作者与版权信息。
- `loong_gui_sim/licenses/`：仿真 SDK、driver、关节映射和 MuJoCo / GLFW 等第三方材料。
- `licenses/loong_gui_ctrl/`：沿用输入包的目录命名，保存 GUI 使用的 Python、PyQt5 / Qt、PyInstaller 等材料；目录存在不代表本次交付机器人控制台。
- `models/`、机器人网格、参考动作及演示素材：保留输入内容及已有声明；尚未在输入包中发现覆盖全部资源的独立再分发授权说明，维护者应补齐来源与授权范围。

## GUI 第三方组件

GUI 使用 PyQt5 / Qt 等第三方组件。输入包包含 PyQt5 的 GPL v3 许可文本，相关材料保留在 `licenses/loong_gui_ctrl/`。适用于本项目的授权依据及分发条件尚待核实。

参考：[PyQt 官方许可说明](https://riverbankcomputing.com/software/pyqt/)、[官方许可 FAQ](https://riverbankcomputing.com/commercial/license-faq)。此项是基于当前交付材料的发布检查，不替代对项目实际授权情况的确认。
