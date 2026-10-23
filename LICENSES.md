# 组件许可与来源

本项目是多个组件的整合发布，不使用单一许可证覆盖所有内容。

- `mjlab_amp_ovision/`：保留 Apache-2.0 许可与源码版权声明，基于 mjlab，强化学习依赖 rsl_rl。
- `loong_gui_sim/runtime/license`：原 SDK 的许可文本，保留原作者与版权信息。
- `loong_gui_sim/licenses/`：仿真 SDK、driver、关节映射和 MuJoCo / GLFW 等第三方材料。
- `licenses/loong_gui_ctrl/`：沿用输入包的目录命名，保存 GUI 使用的 Python、PyQt5 / Qt、PyInstaller 等材料；目录存在不代表本次交付机器人控制台。
- `models/`、机器人网格、参考动作及演示素材：保留输入内容及已有声明；尚未在输入包中发现覆盖全部资源的独立再分发授权说明，维护者应补齐来源与授权范围。

## GUI 发布前待核实

当前输入包的说明明确使用 GPL v3 版 PyQt5，UI 业务模块以预编译程序交付，没有对应 UI 源码和构建材料，也没有附带可确认的商业许可依据。因此，现有许可文件不足以确认这一 GUI 二进制的公开分发条件已经完整满足。

维护者已明确 GUI 只交付二进制、不公开业务源码，本仓库不包含 GUI 私有源码。隐藏源码或将 Python 编译成二进制不会自动解除第三方许可义务；GUI 自身的闭源交付声明也不改变 PyQt 的许可条件。

按当前闭源交付要求，需核实适用的 PyQt 商业授权及对应 Qt 组件的分发条件；若不具备相应授权，则需调整依赖和打包方式后重新检查。单纯购买许可是否足以覆盖现有构建，还需按官方条款确认版本、开发及分发条件。仅保留一份 GPL 文本不能替代这一核实。

参考：[PyQt 官方许可说明](https://riverbankcomputing.com/software/pyqt/)、[官方许可 FAQ](https://riverbankcomputing.com/commercial/license-faq)。此项是基于当前交付材料的发布检查，不替代对项目实际授权情况的确认。
