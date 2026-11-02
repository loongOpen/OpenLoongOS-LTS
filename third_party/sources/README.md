# GUI 第三方库源码

本目录随 GUI 分发 PySide6 / Shiboken6 及所用 Qt 模块 **6.11.3** 的官方源码归档。
具体来源、大小、SHA-256 见 [SOURCES.json](SOURCES.json)。归档没有本地修改，包含上游构建文件、版权声明及各子组件许可。

| 归档 | 对应交付内容 |
|---|---|
| pyside-setup | PySide6、Shiboken6 绑定与支持代码 |
| qtbase | Core、Gui、Widgets、OpenGL、Network、DBus、Test、XcbQpa 及相关平台插件 |
| qtsvg | Svg、SVG 图像与图标插件 |
| qtimageformats | 附加图像格式插件 |

归档通过 Git LFS 分发，需执行 `git lfs pull` 下载实际文件。不能只转发 LFS 指针或只复制 GUI 可执行文件；交付完整目录和这些许可、源码材料。

各源码归档保留上游许可证和版权声明。实际运行文件清单见 `licenses/gui/COMPONENTS.json`，适用范围见根目录 `LICENSES.md`。
