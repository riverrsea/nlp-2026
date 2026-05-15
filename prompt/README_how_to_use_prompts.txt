这些文件是拆分后的 Codex 提示词。建议按以下顺序逐个发送给 Codex：

1. 00_project_overview.txt
用于给 Codex 说明项目背景、数据集、整体目标和总要求。

2. 01_stage1_data_exploration.txt
让 Codex 先完成第一阶段：数据读取、清洗、统计、划分 train/val/test。

3. 02_stage2_engineering_framework.txt
让 Codex 在搭模型之前先构建整体工程框架，包括 src/main.py、cfg/*.yaml、scripts/*.sh、trainers 目录等。

4. 03_stage2_model_building.txt
让 Codex 在工程框架下实现模型，包括 TF-IDF + SVM、TextCNN-Random、TextCNN-Pretrained、BERT、Prompt Learning 接口。

5. 04_stage3_training_validation_testing.txt
让 Codex 完成统一训练、验证、测试、模型对比、混淆矩阵和错误案例分析。

推荐使用方式：
- 不要一次性把所有提示词都发给 Codex。
- 先发 00 和 01，确认数据处理能跑通。
- 数据处理成功后，再发 02。
- 工程框架搭好后，再发 03。
- 模型能运行后，再发 04。
- 每个阶段结束后，都检查对应验收标准。
