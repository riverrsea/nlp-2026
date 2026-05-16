# 项目总览（Project Overview）

## 范围说明
本文档用于说明当前仓库中各文件/目录的作用，排除以下内容：

- `.gitignore` 中被忽略的文件或目录
- `data/` 原始语料目录

## 根目录文件
- `README.md`：项目主说明文档，包含三阶段流程、运行命令和输出说明。
- `requirements.txt`：项目依赖列表。
- `.gitignore`：忽略规则（如虚拟环境、checkpoint、缓存目录等）。
- `project_overview.md`：当前这份中文总览文档。

## 配置目录（`cfg/`）
- `cfg/data_prepare.yaml`：第一阶段数据准备配置（路径、划分比例、绘图参数、编码候选等）。
- `cfg/svm.yaml`：TF-IDF + SVM 实验配置。
- `cfg/textcnn_random.yaml`：TextCNN-Random 实验配置。
- `cfg/textcnn_pretrained.yaml`：TextCNN-Pretrained 实验配置。
- `cfg/bert.yaml`：BERT Fine-tuning 实验配置。
- `cfg/prompt_bert.yaml`：Prompt-BERT 实验配置（模板、verbalizer、few-shot 参数等）。
- `cfg/prompt_gpt.yaml`：GPT Prompt 接口配置。

## 脚本目录（`scripts/`）
- `scripts/run_data_prepare.sh`：运行第一阶段数据准备。
- `scripts/run_stage1_data_prepare.sh`：第一阶段兼容启动脚本（别名入口）。
- `scripts/run_svm.sh`：运行 SVM 训练/验证/测试流程。
- `scripts/run_textcnn_random.sh`：运行 TextCNN-Random。
- `scripts/run_textcnn_pretrained.sh`：运行 TextCNN-Pretrained（需要 `PRETRAINED_PATH`）。
- `scripts/run_bert.sh`：运行 BERT Fine-tuning。
- `scripts/run_prompt_bert.sh`：运行 Prompt-BERT，支持 `FEW_SHOT_K` 和本地模型路径。
- `scripts/run_prompt_gpt.sh`：运行 GPT Prompt 接口流程（占位接口）。
- `scripts/run_stage3_analysis.sh`：生成第三阶段总产物（模型对比表、混淆矩阵、错误案例）。
- `scripts/test_model.sh`：统一的测试模式脚本（`mode=test` + checkpoint）。

## 源码目录（`src/`）
- `src/__init__.py`：包标记文件。
- `src/main.py`：统一命令行入口，负责加载配置、准备运行环境、分发 trainer。
- `src/config.py`：配置加载与合并、默认参数、路径归一化、标签映射注入。
- `src/utils.py`：通用工具函数（日志、文件读写、绘图、随机种子、辅助函数等）。
- `src/data_prepare.py`：第一阶段主流程（读取、清洗、统计、划分、保存 CSV/JSON/图表）。
- `src/dataset.py`：分词、词表、数据集类、BERT tokenizer 接口、预训练词向量加载。
- `src/evaluate.py`：评估指标与报告计算（accuracy、macro 指标、混淆矩阵等）。
- `src/stage3_report.py`：第三阶段汇总脚本（模型对比表、混淆矩阵图、case study）。

### 模型目录（`src/models/`）
- `src/models/__init__.py`：模型包标记。
- `src/models/textcnn.py`：TextCNN 模型结构定义。

### 训练器目录（`src/trainers/`）
- `src/trainers/__init__.py`：trainer 注册与 `model_name` 到 trainer 的分发。
- `src/trainers/base_trainer.py`：trainer 公共基类（数据读取、结果保存、指标包装、few-shot 采样）。
- `src/trainers/svm_trainer.py`：SVM 训练/测试流程。
- `src/trainers/textcnn_trainer.py`：TextCNN（random/pretrained）训练/测试流程。
- `src/trainers/bert_trainer.py`：BERT Fine-tuning 训练/测试流程。
- `src/trainers/prompt_bert_trainer.py`：Prompt-BERT zero-shot / few-shot 推理流程。
- `src/trainers/prompt_gpt_trainer.py`：GPT Prompt 接口占位流程。

## 任务提示目录（`prompt/`）
- `prompt/00_project_overview.txt`：项目总要求与阶段顺序。
- `prompt/01_stage1_data_exploration.txt`：第一阶段任务要求。
- `prompt/02_stage2_engineering_framework.txt`：第二阶段工程框架要求。
- `prompt/03_stage2_model_building.txt`：第二阶段模型实现要求。
- `prompt/04_stage3_training_validation_testing.txt`：第三阶段训练验证测试与分析要求。
- `prompt/README_how_to_use_prompts.txt`：prompt 文件使用说明。

## 字体目录（`font/`）
- `font/SIMFANG.TTF`：第一阶段类别分布图使用的中文字体文件。

## 运行产物目录（`outputs/`）
`outputs/` 是运行脚本后生成的结果目录。

### 数据划分（`outputs/data/`）
- `outputs/data/all_data.csv`：清洗后的全量数据表。
- `outputs/data/train.csv`：训练集。
- `outputs/data/val.csv`：验证集。
- `outputs/data/test.csv`：测试集。

### 词向量（`outputs/embeddings/`）
- `outputs/embeddings/textcnn_pretrained_vocab_only.txt`：按任务词表裁剪后的词向量文件。
- `outputs/embeddings/textcnn_pretrained_vocab_only_stats.json`：裁剪统计信息（覆盖率等）。

### 图表（`outputs/figures/`）
- `outputs/figures/class_distribution.png`：第一阶段类别分布图。
- `outputs/figures/length_distribution.png`：第一阶段文本长度分布图。
- `outputs/figures/textcnn_random_training_curve.png`：TextCNN-Random 综合训练曲线。
- `outputs/figures/textcnn_random_loss.png`：TextCNN-Random 验证损失曲线。
- `outputs/figures/textcnn_random_accuracy.png`：TextCNN-Random 验证准确率曲线。
- `outputs/figures/textcnn_pretrained_training_curve.png`：TextCNN-Pretrained 综合训练曲线。
- `outputs/figures/textcnn_pretrained_loss.png`：TextCNN-Pretrained 验证损失曲线。
- `outputs/figures/textcnn_pretrained_accuracy.png`：TextCNN-Pretrained 验证准确率曲线。
- `outputs/figures/bert_training_curve.png`：BERT 综合训练曲线。
- `outputs/figures/bert_loss.png`：BERT 验证损失曲线。
- `outputs/figures/bert_accuracy.png`：BERT 验证准确率曲线。
- `outputs/figures/confusion_matrix.png`：第三阶段最佳模型混淆矩阵图。

### 预测结果（`outputs/predictions/`）
- `outputs/predictions/svm_predictions.csv`：SVM 测试集预测。
- `outputs/predictions/textcnn_random_predictions.csv`：TextCNN-Random 测试集预测。
- `outputs/predictions/textcnn_pretrained_predictions.csv`：TextCNN-Pretrained 测试集预测。
- `outputs/predictions/bert_predictions.csv`：BERT 测试集预测。
- `outputs/predictions/prompt_bert_zero_shot_predictions.csv`：Prompt-BERT zero-shot 预测。
- `outputs/predictions/prompt_bert_few_shot_2_predictions.csv`：Prompt-BERT few-shot（k=2）预测。

### 指标与报告（`outputs/results/`）
- `outputs/results/data_statistics.json`：第一阶段统计结果。
- `outputs/results/svm_results.json`：SVM 主结果。
- `outputs/results/svm_train_results.json`：SVM 训练模式结果快照。
- `outputs/results/svm_test_results.json`：SVM 测试模式结果快照。
- `outputs/results/textcnn_random_results.json`：TextCNN-Random 主结果。
- `outputs/results/textcnn_random_train_results.json`：TextCNN-Random 训练模式结果快照。
- `outputs/results/textcnn_random_history.json`：TextCNN-Random epoch 历史。
- `outputs/results/textcnn_pretrained_results.json`：TextCNN-Pretrained 主结果。
- `outputs/results/textcnn_pretrained_train_results.json`：TextCNN-Pretrained 训练模式结果快照。
- `outputs/results/textcnn_pretrained_history.json`：TextCNN-Pretrained epoch 历史。
- `outputs/results/bert_results.json`：BERT 主结果。
- `outputs/results/bert_train_results.json`：BERT 训练模式结果快照。
- `outputs/results/bert_history.json`：BERT epoch 历史。
- `outputs/results/prompt_bert_zero_shot_results.json`：Prompt-BERT zero-shot 主结果。
- `outputs/results/prompt_bert_zero_shot_train_results.json`：Prompt-BERT zero-shot 训练模式结果快照。
- `outputs/results/prompt_bert_few_shot_1_results.json`：Prompt-BERT few-shot（k=1）历史结果文件。
- `outputs/results/prompt_bert_few_shot_1_train_results.json`：Prompt-BERT few-shot（k=1）历史训练快照。
- `outputs/results/all_model_comparison.csv`：第三阶段模型对比总表。
- `outputs/results/case_study.md`：第三阶段错误案例分析文档。
