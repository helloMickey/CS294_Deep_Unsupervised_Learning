# CS294_Deep_Unsupervised_Learning

[UC Berkeley CS294 Deep Unsupervised Learning Spring 2020](https://sites.google.com/view/berkeley-cs294-158-sp20/) 的 demo 和 homework。
原Repo：[deepul](https://github.com/rll/deepul)， 
先挖个坑，后面再填，对原项目的代码重新组织下使其：
- 可以在国内环境下运行（部分资源是通过指令下载的，不能访问外网则会报错）
- 整理为jupyter和pycharm两种IDE内都适用（.ipynb文件不好调试，要深入理解代码还是打断点方便）
- 代码增加更多的注释
- 学习笔记

使用pycharm的原因：
- pycharm中不支持远程的jupyter调试！
注意：pycharm中的scientific mode在使用远程解释器时不好用，matplotlib的图会莫名其妙出不来（有时候sciview中就会有图，[相关问题的讨论](https://youtrack.jetbrains.com/issue/PY-32668?_ga=2.39387683.1621664442.1590051461-775331244.1589933127)，最终的解决方法在"Edit Configuration"中取消勾选"Run With Python Console"）。所以建议用jupyter运行去看图，用pycharm设断点运行来理解代码。

pycharm中matpoltlib画出的图显示在sciview中：
![](images\sciview.png)