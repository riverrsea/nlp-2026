# NLP 期末大作业

本仓库基于 `data/` 目录中的中文文本分类数据集完成课程实验。当前已完成第一阶段：数据探索与表征，能够从原始类别文件夹中读取文本、完成基础清洗、统计分析、分层划分，并为后续 TF-IDF + SVM、TextCNN、BERT 模型准备基础数据接口。

## 第一阶段运行方式

推荐在项目根目录执行：

```bash
python src/data_prepare.py
```

也可以通过脚本执行：

```bash
bash scripts/run_stage1_data_prepare.sh
```

如需覆盖默认配置，可使用命令行参数：

```bash
python src/data_prepare.py \
  --config cfg/data_prepare.yaml \
  --data-dir data \
  --output-dir outputs \
  --seed 42 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

## 第一阶段输出文件

运行完成后会自动创建 `outputs/` 及其子目录，并生成以下文件：

- `outputs/data/all_data.csv`
  - 全量样本表，包含 `text`、`label`、`label_id`、`file_path`。
- `outputs/data/train.csv`
  - 按 8:1:1 分层划分后的训练集。
- `outputs/data/val.csv`
  - 按 8:1:1 分层划分后的验证集。项目统一使用 `val` 命名，不使用 `dev`。
- `outputs/data/test.csv`
  - 按 8:1:1 分层划分后的测试集。
- `outputs/results/data_statistics.json`
  - 数据统计结果，包含总样本数、类别数、类别分布、文本长度统计、`label2id`、`id2label`、各数据划分统计信息。
- `outputs/figures/class_distribution.png`
  - 类别样本数量柱状图。
- `outputs/figures/length_distribution.png`
  - 文本长度分布图。

## 第一阶段实现内容

- 兼容读取 `.txt` 和 `.TXT` 文件。
- 优先按 `utf-8` 读取，失败后依次尝试 `gb18030`、`gbk`。
- 对无法读取的文件记录 warning，不让程序整体崩溃。
- 执行基础清洗：处理多余空行、连续空格、制表符、异常控制字符、首尾空白。
- 按固定映射建立标签：

```python
label2id = {
    "art": 0,
    "computer": 1,
    "economy": 2,
    "transportation": 3,
    "education": 4,
    "environment": 5,
    "sports": 6,
    "military": 7,
    "politics": 8,
    "medicine": 9,
}
```

- 为后续模型预留了以下接口：
  - `src/dataset.py` 中的 `tokenize_with_jieba`，用于 TF-IDF + SVM。
  - `src/dataset.py` 中的词表构建、token id 转换、padding、truncation、`TextCNNDataset`，用于 TextCNN。
  - `src/dataset.py` 中的 `load_pretrained_word_vectors`，为后续随机 Embedding / 预训练词向量对比预留接口。
  - `src/dataset.py` 中的 `load_bert_tokenizer` 和 `prepare_bert_inputs`，用于 BERT 输入准备。

## 依赖说明

项目依赖已写入 `requirements.txt`。第一阶段的数据准备脚本可直接运行；若要继续第二、三阶段的模型训练，请先安装依赖：

```bash
pip install -r requirements.txt
```
