# FIPread 所需库清单

本清单记录了 FIPread 运行与打包所需的全部第三方 Python 库，供在新机器上重建环境时参考。

## 一、运行依赖（必须安装）

这些库在源码运行（`python run.py`）和打包后的 exe 中都会被用到。

| 库 | 版本 | 用途 |
| --- | --- | --- |
| numpy | 1.26.4 | 数组运算与波形数据处理 |
| scipy | 1.13.1 | 滤波、welch PSD、统计量、wav 读写 |
| PyQt5 | 5.15.10 | GUI 界面框架 |
| pyqtgraph | 0.13.7 | 波形与 t-f 图绘制 |
| nptdms | 1.10.0 | TDMS 文件读写 |
| pandas | 2.0.3 | 数据处理与特征统计 |
| joblib | 1.4.2 | SVM 模型加载 |
| scikit-learn | 1.6.1 | SVM 预测 Pipeline |
| matplotlib | 3.9.2 | t-f 图颜色映射（colormap） |

## 二、打包依赖（仅打包 exe 时需要）

| 库 | 版本 | 用途 |
| --- | --- | --- |
| pyinstaller | 6.9.0 | 将程序打包为单文件 exe |
| Pillow | 11.1.0 | 将 `logo.png` 转换为 exe 图标（.ico） |

## 三、版本对照环境

以上版本在开发环境（conda 环境 `LZdataread39`，Python 3.9.19，Windows）中验证通过。

## 四、批量安装

运行依赖：

```powershell
pip install -r requirements.txt
```

打包依赖：

```powershell
pip install -r requirements-build.txt
```

一次性全部安装：

```powershell
pip install -r requirements.txt -r requirements-build.txt
```
