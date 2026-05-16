# NLP 期末大作业

本仓库基于 `data/` 目录中的中文文本分类数据集完成课程实验，当前已经完成：

- 第一阶段：数据探索与表征
- 第二阶段：模型建模
- 第三阶段：训练、验证与测试分析

## 环境准备

推荐使用当前目录下的虚拟环境：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

如果还没有虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 工程结构

```text
cfg/
├── data_prepare.yaml
├── svm.yaml
├── textcnn_random.yaml
├── textcnn_pretrained.yaml
├── bert.yaml
├── prompt_bert.yaml
└── prompt_gpt.yaml

scripts/
├── run_data_prepare.sh
├── run_stage1_data_prepare.sh
├── run_stage3_analysis.sh
├── run_svm.sh
├── run_textcnn_random.sh
├── run_textcnn_pretrained.sh
├── run_bert.sh
├── run_prompt_bert.sh
├── run_prompt_gpt.sh
└── test_model.sh

src/
├── main.py
├── config.py
├── utils.py
├── data_prepare.py
├── dataset.py
├── evaluate.py
├── trainers/
│   ├── base_trainer.py
│   ├── svm_trainer.py
│   ├── textcnn_trainer.py
│   ├── bert_trainer.py
│   ├── prompt_bert_trainer.py
│   └── prompt_gpt_trainer.py
└── models/
    └── textcnn.py
```

## 第一阶段：数据探索与表征

运行：

```bash
python src/data_prepare.py
```

或：

```bash
bash scripts/run_data_prepare.sh
```

第一阶段会生成：

- `outputs/data/all_data.csv`
- `outputs/data/train.csv`
- `outputs/data/val.csv`
- `outputs/data/test.csv`
- `outputs/results/data_statistics.json`
- `outputs/figures/class_distribution.png`
- `outputs/figures/length_distribution.png`

其中 `all_data.csv`、`train.csv`、`val.csv`、`test.csv` 都包含：

- `text`
- `label`
- `label_id`
- `file_path`

## 统一入口

统一入口为 `src/main.py`：

```bash
python src/main.py --config cfg/svm.yaml --mode train
```

当前支持的主要参数：

- `--config`：配置文件路径，例如 `cfg/textcnn_random.yaml`
- `--mode`：`train` 或 `test`
- `--data_dir`：原始数据目录，默认 `data`
- `--output_dir`：输出目录，默认 `outputs`
- `--device`：运行设备，默认 `auto`
- `--seed`：随机种子
- `--checkpoint`：测试模式下加载的 checkpoint 路径
- `--pretrained_path`：TextCNN-Pretrained 的外部词向量路径
- `--model_name_or_path`：BERT 或 Prompt-BERT 的预训练模型名称或本地路径
- `--few_shot_k`：Prompt 方法每类抽取的 few-shot 样本数
- `--log_level`：日志级别

## cfg 配置方式

每个实验都通过 `cfg/*.yaml` 管理超参数，典型结构包括：

- `experiment_name`
- `model_name`
- `data`
- `model`
- `train`
- `eval`
- `prompt`
- `logging`

例如：

```yaml
experiment_name: textcnn_random
model_name: textcnn
embedding_type: random

data:
  train_file: outputs/data/train.csv
  val_file: outputs/data/val.csv
  test_file: outputs/data/test.csv
  max_len: 512
```

## 第二阶段：模型建模

当前已经接入以下模型或接口：

### 1. TF-IDF + SVM

运行：

```bash
bash scripts/run_svm.sh
```

或：

```bash
python src/main.py --config cfg/svm.yaml --mode train --output_dir outputs
```

功能：

- 使用 `jieba` 分词
- 使用 `TfidfVectorizer` 提取特征
- 使用 `LinearSVC` 或 `SVC`
- 支持 `max_features`、`ngram_range`、`C` 等配置
- 保存 checkpoint、测试结果和预测文件

主要输出：

- `outputs/results/svm_results.json`
- `outputs/predictions/svm_predictions.csv`
- `outputs/checkpoints/svm/svm_best.ckpt`

### 2. TextCNN-Random

运行：

```bash
bash scripts/run_textcnn_random.sh
```

或：

```bash
python src/main.py --config cfg/textcnn_random.yaml --mode train --output_dir outputs
```

功能：

- 中文分词
- 词表构建
- 随机初始化 Embedding
- 多尺度卷积核 `3/4/5`
- Max Pooling、Dropout、Linear 分类层
- 在 `val.csv` 上按 `macro_f1` 选择最佳模型

主要输出：

- `outputs/results/textcnn_random_results.json`
- `outputs/predictions/textcnn_random_predictions.csv`
- `outputs/checkpoints/textcnn_random/textcnn_random_best.ckpt`
- `outputs/results/textcnn_random_history.json`
- `outputs/figures/textcnn_random_training_curve.png`

### 3. TextCNN-Pretrained

运行：

```bash
PRETRAINED_PATH=path/to/vector.txt bash scripts/run_textcnn_pretrained.sh
```

或：

```bash
python src/main.py \
  --config cfg/textcnn_pretrained.yaml \
  --mode train \
  --output_dir outputs \
  --pretrained_path path/to/vector.txt
```

说明：

- 使用与 `TextCNN-Random` 相同的词表构建与训练流程
- 仅 Embedding 初始化方式不同
- 如果没有提供合法的 `pretrained_path`，程序会给出清晰错误提示
- 支持 Word2Vec / FastText / 腾讯词向量等常见文本格式词向量文件
- 项目已提供裁剪后的任务专用词向量，可直接使用：
  `outputs/embeddings/textcnn_pretrained_vocab_only.txt`
- 对应的裁剪统计信息保存在：
  `outputs/embeddings/textcnn_pretrained_vocab_only_stats.json`

主要输出：

- `outputs/results/textcnn_pretrained_results.json`
- `outputs/predictions/textcnn_pretrained_predictions.csv`
- `outputs/checkpoints/textcnn_pretrained/textcnn_pretrained_best.ckpt`

### 4. BERT Fine-tuning

运行：

```bash
bash scripts/run_bert.sh
```

或：

```bash
python src/main.py \
  --config cfg/bert.yaml \
  --mode train \
  --output_dir outputs \
  --model_name_or_path pretrained_models/bert-base-chinese
```

说明：

- 使用 HuggingFace `AutoTokenizer`
- 使用 `AutoModelForSequenceClassification`
- 默认模型是 `bert-base-chinese`
- 可以切换到 `hfl/chinese-roberta-wwm-ext`
- 如果当前环境不能从 HuggingFace 下载模型，程序会提示你改用本地模型目录

推荐本地模型运行方式：

```bash
python src/main.py \
  --config cfg/bert.yaml \
  --mode train \
  --output_dir outputs \
  --model_name_or_path pretrained_models/bert-base-chinese
```

主要输出：

- `outputs/results/bert_results.json`
- `outputs/predictions/bert_predictions.csv`
- `outputs/checkpoints/bert/bert_best.ckpt`

### 5. BERT Prompt Learning

运行：

```bash
bash scripts/run_prompt_bert.sh
```

或 zero-shot：

```bash
python src/main.py \
  --config cfg/prompt_bert.yaml \
  --mode train \
  --output_dir outputs \
  --model_name_or_path pretrained_models/bert-base-chinese \
  --few_shot_k 0
```

few-shot 示例：

```bash
python src/main.py \
  --config cfg/prompt_bert.yaml \
  --mode train \
  --output_dir outputs \
  --few_shot_k 2 \
  --model_name_or_path pretrained_models/bert-base-chinese
```

说明：

- 使用 Masked Language Model
- Prompt 模板为 `这是一篇关于[MASK]的文章：{text}`
- few-shot 时从 `train.csv` 中每类抽取 `k` 条样本作为紧凑示例
- 如果 HuggingFace 模型不可用，会给出明确提示

主要输出：

- `outputs/results/prompt_bert_zero_shot_results.json`
- `outputs/results/prompt_bert_few_shot_2_results.json`
- `outputs/predictions/prompt_bert_zero_shot_predictions.csv`
- `outputs/predictions/prompt_bert_few_shot_2_predictions.csv`

### 6. GPT Prompt Learning 接口

运行：

```bash
bash scripts/run_prompt_gpt.sh
```

或：

```bash
python src/main.py \
  --config cfg/prompt_gpt.yaml \
  --mode train \
  --output_dir outputs \
  --few_shot_k 2
```

说明：

- 当前实现的是“可扩展接口”，不会直接调用远程 GPT API
- 如果没有本地生成式模型或 API key，程序会输出清晰提示，但不会让整个项目失败
- 会保存占位结果文件和占位预测文件，方便后续接入真实模型

当前输出：

- `outputs/results/prompt_gpt_results.json`
- `outputs/predictions/prompt_gpt_predictions.csv`

接入本地模型或 API 的建议方式：

1. 在 `src/trainers/prompt_gpt_trainer.py` 中补充真实推理函数。
2. 将模型生成的中文类别名映射回固定 10 类标签。
3. 使用 `BaseTrainer.save_predictions()` 和 `BaseTrainer.save_results()` 保存正式结果。

## 测试已训练模型

统一测试脚本：

```bash
bash scripts/test_model.sh cfg/svm.yaml outputs/checkpoints/svm/svm_best.ckpt
```

也可以直接执行：

```bash
python src/main.py \
  --config cfg/textcnn_random.yaml \
  --mode test \
  --output_dir outputs \
  --checkpoint outputs/checkpoints/textcnn_random/textcnn_random_best.ckpt
```

## 第三阶段：训练、验证与测试分析

第三阶段当前已经统一完成以下实验与分析产物：

- `TF-IDF + SVM`
- `TextCNN-Random`
- `TextCNN-Pretrained`
- `BERT Fine-tuning`
- `BERT Prompt Zero-shot`
- `BERT Prompt Few-shot`

`GPT Prompt` 目前仍保留为可扩展接口，第三阶段汇总脚本默认跳过它。

### 推荐运行顺序

1. 数据准备

```bash
bash scripts/run_data_prepare.sh
```

2. 训练传统模型

```bash
bash scripts/run_svm.sh
```

3. 训练 TextCNN-Random

```bash
bash scripts/run_textcnn_random.sh
```

4. 训练 TextCNN-Pretrained

```bash
PRETRAINED_PATH=outputs/embeddings/textcnn_pretrained_vocab_only.txt bash scripts/run_textcnn_pretrained.sh
```

5. 如果具备本地 BERT 模型，再运行 BERT

```bash
bash scripts/run_bert.sh
```

6. 运行 Prompt-BERT Zero-shot

```bash
MODEL_NAME_OR_PATH=pretrained_models/bert-base-chinese FEW_SHOT_K=0 bash scripts/run_prompt_bert.sh
```

7. 运行 Prompt-BERT Few-shot

```bash
MODEL_NAME_OR_PATH=pretrained_models/bert-base-chinese FEW_SHOT_K=2 bash scripts/run_prompt_bert.sh
```

8. 生成第三阶段最终汇总材料

```bash
bash scripts/run_stage3_analysis.sh
```

### 第三阶段输出

当前已生成或会生成：

- `outputs/results/all_model_comparison.csv`
- `outputs/figures/confusion_matrix.png`
- `outputs/results/case_study.md`
- `outputs/figures/textcnn_random_loss.png`
- `outputs/figures/textcnn_random_accuracy.png`
- `outputs/figures/textcnn_pretrained_loss.png`
- `outputs/figures/textcnn_pretrained_accuracy.png`

模型结果文件：

- `outputs/results/svm_results.json`
- `outputs/results/textcnn_random_results.json`
- `outputs/results/textcnn_pretrained_results.json`
- `outputs/results/bert_results.json`
- `outputs/results/prompt_bert_zero_shot_results.json`
- `outputs/results/prompt_bert_few_shot_2_results.json`

模型预测文件：

- `outputs/predictions/svm_predictions.csv`
- `outputs/predictions/textcnn_random_predictions.csv`
- `outputs/predictions/textcnn_pretrained_predictions.csv`
- `outputs/predictions/bert_predictions.csv`
- `outputs/predictions/prompt_bert_zero_shot_predictions.csv`
- `outputs/predictions/prompt_bert_few_shot_2_predictions.csv`

### 如何查看最终结果

最终模型对比表：

```bash
sed -n '1,20p' outputs/results/all_model_comparison.csv
```

也可以直接打开：

- [all_model_comparison.csv](outputs/results/all_model_comparison.csv)
- [confusion_matrix.png](outputs/figures/confusion_matrix.png)
- [case_study.md](outputs/results/case_study.md)

### 常见问题

1. HuggingFace 模型无法下载

请改用本地模型目录：

```bash
python src/main.py \
  --config cfg/bert.yaml \
  --mode train \
  --output_dir outputs \
  --model_name_or_path pretrained_models/bert-base-chinese
```

2. GPT Prompt 为什么没有参与最终比较

当前第三阶段按实验需要优先完成可复现的判别式模型和 BERT 代码路径，`GPT Prompt` 仍保留接口，但默认不纳入这次最终实验统计。

## 当前说明

- SVM 已完成可训练与可测试版本。
- TextCNN-Random 已完成可训练与可测试版本。
- TextCNN-Pretrained 已完成接口与训练逻辑，项目中已提供可直接使用的裁剪词向量。
- 第三阶段的最终比较表、混淆矩阵和 case study 已可直接生成。
- BERT Fine-tuning 已完成训练与测试流程，默认可使用本地 `pretrained_models/bert-base-chinese`。
- Prompt-BERT 已完成 zero-shot / few-shot 实验流程，并已接入第三阶段汇总。
- GPT Prompt 当前为可扩展接口，不会因为缺少 API 或本地模型而导致整个项目失败。
