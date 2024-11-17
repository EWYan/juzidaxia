# venv实践
---
> 🪶date: 2024/11/17  

## 1. 关于python虚拟环境
创建一个虚拟环境，并安装python包。用于隔离不同python版本，不同python包的依赖。

## 2. 创建虚拟环境 🌀 
- 先看下本地pyhton版本以及installed packages
```bash
python --version
```
```txt title="prompt logs"
Python 3.12.1
```
```bash
pip list
```
```txt title="prompt logs"
Package    Version
---------- ---------
Package                                   Version
----------------------------------------- ------------
Babel                                     2.14.0
blinker                                   1.9.0
certifi                                   2023.11.17
charset-normalizer                        3.3.2
click                                     8.1.7
colorama                                  0.4.6
dirtyjson                                 1.0.8
Flask                                     2.3.2
ghp-import                                2.1.0
gitdb                                     4.0.11
GitPython                                 3.1.40
etc...
```
- 创建虚拟环境
```bash
python -m venv venv_test
```
```txt title="prompt logs"
PS C:\lab\tryvenv> ls .\venv_test\


    Directory: C:\lab\tryvenv\venv_test


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        11/17/2024   5:06 PM                Include
d-----        11/17/2024   5:06 PM                Lib
d-----        11/17/2024   5:06 PM                Scripts
-a----        11/17/2024   5:06 PM            183 pyvenv.cfg

```
- 激活虚拟环境

在windows下激活虚拟环境, 直接执行Scripts目录下的Activate.ps1脚本(PowerShell)
```bash
.\venv_test\Scripts\Activate.ps1
```
激活后，可以看到pip版本已经变成了虚拟环境中的版本，可以看出当前环境下使用的pip是虚拟环境中的pip；
查看虚拟环境中的packages, 可以看到当前虚拟环境中安装的packages只有pip
```txt title="prompt logs"
(venv_test) PS C:\lab\tryvenv> pip --version
pip 23.2.1 from C:\lab\tryvenv\venv_test\Lib\site-packages\pip (python 3.12)

(venv_test) PS C:\lab\tryvenv> pip list
Package Version
------- -------
pip     23.2.1

```
当前环境变量中多了几个变量，一个是VIRTUAL_ENV，一个是VIRTUAL_ENV_PROMPT，分别表示虚拟环境的路径，虚拟环境名称。
同时将虚拟环境中的python路径加入到了PATH中。
```txt title="prompt logs" hl_lines="5 7 10-11"
(venv_test) PS C:\lab\tryvenv> gci Env:

Name                           Value
----                           -----
_OLD_VIRTUAL_PATH              c:\Users\AppData\Local\Programs\cursor\resources\app\bin;
OS                             Windows_NT
Path                           C:\lab\tryvenv\venv_test\Scripts;c:\Users\AppData\Local\Programs;...
SystemDrive                    C:
SystemRoot                     C:\WINDOWS
VIRTUAL_ENV                    C:\lab\tryvenv\venv_test
VIRTUAL_ENV_PROMPT             venv_test
VSCODE_GIT_ASKPASS_EXTRA_ARGS
...

```
之后在虚拟环境中安装packages，如安装requests包, 可以看到安装的packages在虚拟环境中的Lib\site-packages目录下
```bash
pip install requests
ls .\venv_test\Lib\site-packages
```
```txt title="prompt logs" linenums="1" hl_lines="6-11 14-17"

    Directory: C:\lab\tryvenv\venv_test\Lib\site-packages


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        11/17/2024   5:29 PM                certifi
d-----        11/17/2024   5:29 PM                certifi-2024.8.30.dist-info
d-----        11/17/2024   5:29 PM                charset_normalizer
d-----        11/17/2024   5:29 PM                charset_normalizer-3.4.0.dist-info
d-----        11/17/2024   5:29 PM                idna
d-----        11/17/2024   5:29 PM                idna-3.10.dist-info
d-----        11/17/2024   5:06 PM                pip
d-----        11/17/2024   5:06 PM                pip-23.2.1.dist-info
d-----        11/17/2024   5:29 PM                requests
d-----        11/17/2024   5:29 PM                requests-2.32.3.dist-info
d-----        11/17/2024   5:29 PM                urllib3
d-----        11/17/2024   5:29 PM                urllib3-2.2.3.dist-info
```
## 3. 退出虚拟环境 🌠
在虚拟环境中，执行deactivate命令，退出虚拟环境
```bash
deactivate
```
## 4. 在虚拟环境中离线安装packages 📦
例如当前使用虚拟环境的主机是offline，而offline主机上已经保存了所需安装的packages的whl文件(例如pandas)，通过如下方式在虚拟环境中安装packages：
```bash
pip install pandas --no-index --find-links=C:\lab\tryvenv\whls
pip list
```
```txt title="prompt logs" linenums="1" hl_lines="7"
Package    Version
---------- ---------
certifi            2024.8.30
charset-normalizer 3.4.0
idna               3.10
numpy              2.1.3
pandas             2.2.3
pip                23.2.1
python-dateutil    2.9.0.post0
pytz               2024.2
requests           2.32.3
six                1.16.0
tzdata             2024.2
urllib3            2.2.3
```

## 5. 在虚拟环境中添加非默认Lib/site-packages引用 📂
只需要在虚拟环境的Lib/site-packages目录下创建一个后缀为.pth的文件，文件内容为实际的packages目录路径即可。

## 6. 设置vscode的python解释器为虚拟环境中的python 🔍
如果vscode的python解释器为虚拟环境中的python，则vscode会使用虚拟环境中的python解释器来解释python代码，方便基于vscode进行python的debugger，实际设置步骤如下：

- 使用快捷键Ctrl+Shift+P，打开命令面板，输入python，选择设置python解释器
- 选择python路径为虚拟环境中的python路径，例如C:\lab\tryvenv\venv_test\python.exe

参考vscode的[官方文档](https://code.visualstudio.com/docs/python/environments)即可。


## 7. 参考资料
- [Python虚拟环境](https://docs.python.org/3/library/venv.html)
- [Embedded Python虚拟环境设置](https://virtualenv.pypa.io/en/latest/user_guide.html)
---