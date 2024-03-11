这是一个提取模型不同视角关键数据的基于pyopengl的python脚本，在尝试过moderngl、open3d、pymesh和pytorch3d等诸多python库之后做出选择，因为pyopengl最切近opengl原生，灵活性更高，扩展性更强。

## 加载依赖
python -m pip install PyOpenGL PyOpenGL_accelerate

## 问题：使用 pip 安装了 PyOpenGL 包，然后运行程序，结果提示：OpenGL Q.error.NullF unction Error: Attempt to call an undefined
function glutinit
原因：使用 pip 安装的 OpenGL 包是 32 位，与64 位电脑不匹配，故出现此错误。
解决办法：pip 不能在线安装 64 位的 OpenGL，只能手动下载后安装。