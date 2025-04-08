# 项目名称

## 代码核心功能说明

本项目旨在实现文本分析功能，支持两种特征提取模式：高频词和TF-IDF。以下是核心功能说明：

1. **文本预处理**：包括分词、去除停用词、词干提取等。
2. **特征提取**：
   - **高频词模式**：提取文本中出现频率最高的词汇作为特征。
   - **TF-IDF模式**：使用TF-IDF算法计算词汇的重要性作为特征。
3. **结果输出**：生成特征向量并进行可视化展示。

### 特征模式切换方法

项目支持两种特征提取模式的切换，具体方法如下：

#### 方法一：通过配置文件切换

1. 在项目根目录下找到`config.json`文件。
2. 修改`feature_mode`字段的值：
   ```json
   {
     "feature_mode": "tfidf"  // 或 "frequency"
   }
   ```
3. 保存文件并重新运行程序。

#### 方法二：通过命令行参数切换

运行程序时，可以通过`--mode`参数指定特征模式：
```bash
python main.py --mode tfidf  # 或 frequency
```

### 数学公式说明

TF-IDF的计算公式如下：
$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$
其中：
- $\text{TF}(t, d)$ 表示词 $t$ 在文档 $d$ 中的词频。
- $\text{IDF}(t)$ 表示词 $t$ 的逆文档频率：
$$
\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right)
$$
其中 $N$ 是总文档数，$\text{DF}(t)$ 是包含词 $t$ 的文档数。

## 项目结构

```
project-root/
├── data/
│   ├── input.txt
│   └── stopwords.txt
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── tfidf.py
│   └── frequency.py
├── config.json
└── README.md
```






