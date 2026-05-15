# NLP 期末大作业

本仓库基于 `data/` 目录中的中文文本分类数据集完成课程实验。目前已经完成：

- 第一阶段：数据探索与表征
- 第二阶段第一部分：整体工程框架构建

项目统一使用 `val` 作为验证集命名，不使用 `dev`。

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
│   ├── __init__.py
│   ├── base_trainer.py
│   ├── svm_trainer.py
│   ├── textcnn_trainer.py
│   ├── bert_trainer.py
│   ├── prompt_bert_trainer.py
│   └── prompt_gpt_trainer.py
└── models/
    ├── __init__.py
    └── textcnn.py
```

## 环境准备

推荐使用当前目录下的虚拟环境：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

所有依赖已写入 `requirements.txt`，至少包含：

- `pandas`
- `numpy`
- `scikit-learn`
- `jieba`
- `matplotlib`
- `seaborn`
- `torch`
- `transformers`
- `tqdm`
- `pyyaml`

## 第一阶段：数据探索与表征

推荐执行：

```bash
python src/data_prepare.py
```

也可以使用脚本：

```bash
bash scripts/run_data_prepare.sh
```

运行完成后会自动创建 `outputs/` 及其子目录，并生成：

- `outputs/data/all_data.csv`
- `outputs/data/train.csv`
- `outputs/data/val.csv`
- `outputs/data/test.csv`
- `outputs/results/data_statistics.json`
- `outputs/figures/class_distribution.png`
- `outputs/figures/length_distribution.png`

其中：

- `all_data.csv` 包含 `text`、`label`、`label_id`、`file_path`
- `train.csv`、`val.csv`、`test.csv` 为按 `8:1:1` 分层划分后的数据
- `data_statistics.json` 包含总样本数、类别数、类别分布、长度统计、`label2id`、`id2label`

## 第二阶段：整体工程框架

第二阶段当前已完成统一入口、配置体系、trainer 路由和启动脚本。具体模型训练逻辑会在后续建模阶段继续补齐。

### 统一入口

统一入口为：

```bash
python src/main.py --config cfg/svm.yaml --mode train
```

`src/main.py` 当前支持以下参数：

- `--config`：配置文件路径，例如 `cfg/textcnn_random.yaml`
- `--mode`：运行模式，取值为 `train` 或 `test`
- `--data_dir`：原始数据目录，默认 `data`
- `--output_dir`：输出目录，默认 `outputs`
- `--device`：运行设备，默认 `auto`
- `--seed`：随机种子
- `--checkpoint`：`mode=test` 时加载的 checkpoint 路径
- `--pretrained_path`：TextCNN-Pretrained 使用的预训练词向量路径
- `--model_name_or_path`：BERT 或 Prompt-BERT 使用的预训练模型名称或本地路径
- `--log_level`：日志级别

### cfg 配置方式

每个实验使用独立的 `cfg/*.yaml` 管理超参数。配置文件统一包含以下层级：

- `experiment_name`：实验名
- `model_name`：模型名，当前支持 `svm`、`textcnn`、`bert`、`prompt_bert`、`prompt_gpt`
- `data`：`train_file`、`val_file`、`test_file`、`max_len`
- `model`：模型结构或推理参数
- `train`：训练超参数，例如 `batch_size`、`epochs`、`learning_rate`
- `eval`：评估指标列表
- `logging`：日志级别

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

### scripts 用法

直接运行脚本即可触发对应实验：

```bash
bash scripts/run_svm.sh
bash scripts/run_textcnn_random.sh
bash scripts/run_bert.sh
```

如果是 TextCNN 预训练词向量版本：

```bash
PRETRAINED_PATH=path/to/vector.txt bash scripts/run_textcnn_pretrained.sh
```

如果要测试某个 checkpoint：

```bash
bash scripts/test_model.sh cfg/bert.yaml path/to/checkpoint.ckpt
```

### 当前框架状态

当前 `main.py` 已经能够：

- 读取 YAML 配置并与命令行参数合并
- 设置随机种子
- 自动创建 `outputs/data`、`outputs/figures`、`outputs/results`、`outputs/predictions`、`outputs/checkpoints`
- 根据 `model_name` 路由到对应 trainer
- 区分 `mode=train` 和 `mode=test`
- 在 `mode=test` 下对缺失 checkpoint 给出清晰错误提示

当前 trainer 仍是框架占位实现：

- 会读取 `train.csv`、`val.csv`、`test.csv`
- 会检查路径、记录数据规模
- 会在 `outputs/results/` 下保存框架说明文件
- 会明确提示“具体模型训练逻辑将在后续阶段实现”

这一步的目的，是先把工程入口、配置组织和实验启动方式固定下来，后续建模直接接在这套框架上。
