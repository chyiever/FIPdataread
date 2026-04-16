# sig_mark

`sig_mark` 是基于当前版本 `FIPread` 派生的信号样本标记软件，用于对 `.npz` / `.tdms` 波形文件做滑窗浏览、人工标签标注、初至标注和样本导出。

## 用户说明

### 功能概览

- 左侧参数区，右侧绘图区，布局与 `FIPread` 相近
- 支持读取 `.npz` / `.tdms` 数据文件
- 支持与 `FIPread` 一致的显示滤波预处理
- 右侧绘图区包含：
  - 时域图
  - t-f 图
- 支持设置样本长度、滑动步长、跳转到指定窗口编号
- 支持自定义标签代码并弹出标签按钮
- 支持 `Before`、`Skip`、`Close`
- 支持在保存前用 `Ctrl + Right Click` 标注当前窗口初至
- 当前文件标注到末尾后自动切换到下一个文件

### 当前不包含的功能

与原始 `FIPread` 相比，当前 `sig_mark` 不包含：

- `Plot 2`
- 独立 `PSD` 图
- 音频播放
- 音频导出

### 环境

推荐：

- Windows
- Python 3.9+

安装依赖：

```powershell
pip install numpy scipy PyQt5 pyqtgraph nptdms pandas joblib scikit-learn
```

### 运行

在 `sig_mark` 目录下运行：

```powershell
python .\run.py
```

### 基本使用流程

1. 设置输入目录和导出目录。
2. 从文件列表中选择一个波形文件。
3. 在 `Preprocess` 中设置滤波参数并应用。
4. 在 `Sample Window` 中设置样本长度和步长。
5. 点击 `Show First Window` 开始标注。
6. 在弹出的标记窗口中点击标签按钮，自动保存并切换到下一个窗口。
7. 如需回退，点击 `Before`。
8. 如需跳过当前样本，点击 `Skip`。
9. 如需给当前窗口标注初至，在时域图中执行 `Ctrl + Right Click` 后再保存。

### Label Codes 使用方式

- 左侧编码框可直接按行编辑
- 右侧 `Input` 输入框可追加单个新代码
- 点击 `Add` 将输入框内容加入标签列表
- 点击 `Update Buttons` 刷新弹出标签按钮
- 点击 `Open Label Panel` 打开样本标记弹窗

### 导出规则

每个已标记样本导出为一个 `.npz` 文件。

文件名格式：

```text
类型编码-FIP-采样率-起始时间.npz
```

示例：

```text
F130A-FIP-200K-20260324T100107.350.npz
```

### 重标记规则

同一个窗口永远只保留一个结果。

具体逻辑：

- 保存当前标签前，程序会先删除该窗口先前已导出的旧标签文件
- 然后再保存新的标签文件

因此，当使用 `Before` 回到前一个窗口重新标记时，旧标签会被新标签替换。

## 开发说明

### 目录与入口

- 启动入口：[run.py](E:\codes\FIPread\sig_mark\run.py)
- 主窗口：[src/main_window.py](E:\codes\FIPread\sig_mark\src\main_window.py)
- 数据读写：[src/data_access.py](E:\codes\FIPread\sig_mark\src\data_access.py)
- 预处理与 t-f 计算：[src/processing.py](E:\codes\FIPread\sig_mark\src\processing.py)
- 绘图轴与样式：[src/plotting.py](E:\codes\FIPread\sig_mark\src\plotting.py)

### 当前界面结构

- `Files`
  - 输入目录
  - 导出目录
  - 文件列表
  - 排序
- `Preprocess`
  - 滤波开关
  - 滤波模式
  - `Low Cut / High Cut`
- `Sample Window`
  - `Length / Hop`
  - `Show First Window`
  - `Go To Window`
- `Label Codes`
  - 左侧编码框
  - 右侧 `Input / Add / Update Buttons / Open Label Panel`
- `t-f Display`
  - `Value Scale / Window`
  - `Overlap / Colormap`
  - `Y Min / Y Max`
  - `Color Min / Color Max`

### 数据导出结构

导出的 `npz` 与 `FIPread` 保持兼容，当前包含：

- `phase_data`
- `sample_rate`
- `comm_count`
- `npts`
- `timestamp`
- `starttime`
- `arrival_time`
- `type`
- `data_info`

其中：

- `type` 存储当前标签代码
- `data_info["sample_type"]` 同步保存标签代码
- `arrival_time` 存储当前窗口初至时间（如果已标记）

### 文件命名与覆盖逻辑

命名函数位于 [src/data_access.py](E:\codes\FIPread\sig_mark\src\data_access.py)：

- `build_export_npz_name(...)`
- `build_export_npz_suffix(...)`

保存逻辑位于 [src/main_window.py](E:\codes\FIPread\sig_mark\src\main_window.py) 的 `_label_current_sample(...)`：

- 保存前先按“采样率 + 起始时间”匹配同窗口旧文件
- 删除旧文件
- 再写入当前类型的新文件

### t-f 图实现说明

- t-f 图基于 `scipy.signal.spectrogram`
- 显示前会先把线性频率谱重采样到等间距 `log10(f)` 网格
- 这样可以避免把线性频率矩阵直接错误映射到对数频率轴
- y 轴刻度风格已与当前 `FIPread` 对齐

### 现阶段实现状态

当前版本已经实现：

- 滑窗样本浏览
- 弹窗式标签标注
- 初至标注
- 自动切换到下一窗口/下一文件
- 回退重标并覆盖旧结果
- 与 `FIPread` 风格接近的 t-f 显示与坐标轴

当前版本尚未实现：

- 多人协作标注记录
- 标注索引数据库
- 专门的样本进度管理文件
- 专门的标注结果统计页面
