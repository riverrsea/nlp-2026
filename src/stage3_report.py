from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from dataset import tokenize_with_jieba
from utils import read_csv_rows, save_confusion_matrix_figure, setup_logging, write_csv


LABEL_TO_CHINESE = {
    "art": "艺术",
    "computer": "计算机",
    "economy": "经济",
    "transportation": "交通",
    "education": "教育",
    "environment": "环境",
    "sports": "体育",
    "military": "军事",
    "politics": "政治",
    "medicine": "医药",
}


COMPARISON_ORDER = [
    ("TF-IDF + SVM", "svm_results.json"),
    ("TextCNN-Random", "textcnn_random_results.json"),
    ("TextCNN-Pretrained", "textcnn_pretrained_results.json"),
    ("BERT Fine-tuning", "bert_results.json"),
    ("BERT Prompt Zero-shot", "prompt_bert_zero_shot_results.json"),
    ("BERT Prompt Few-shot", "prompt_bert_few_shot_results.json"),
    ("GPT Prompt Zero-shot", None),
    ("GPT Prompt Few-shot", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stage 3 comparison reports, confusion matrix, and case study."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root output directory containing results, figures, and predictions.",
    )
    parser.add_argument(
        "--skip_gpt",
        action="store_true",
        help="Leave GPT prompt rows empty in the final comparison table.",
    )
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def collect_prompt_result_files(results_dir: Path) -> list[dict[str, Any]]:
    prompt_files = sorted(results_dir.glob("*prompt*results.json"))
    collected: list[dict[str, Any]] = []
    for path in prompt_files:
        data = load_json_if_exists(path)
        if data is None:
            continue
        data["_path"] = str(path)
        collected.append(data)
    return collected


def pick_prompt_result(prompt_results: list[dict[str, Any]], model_name: str, few_shot_k: int) -> dict[str, Any] | None:
    for item in prompt_results:
        if item.get("model_name") != model_name:
            continue
        item_k = int(item.get("few_shot_k", 0))
        if few_shot_k == 0 and item_k == 0:
            return item
        if few_shot_k > 0 and item_k > 0:
            return item
    return None


def load_result_map(results_dir: Path) -> dict[str, dict[str, Any]]:
    result_map: dict[str, dict[str, Any]] = {}
    prompt_results = collect_prompt_result_files(results_dir)

    for display_name, filename in COMPARISON_ORDER:
        if display_name == "BERT Prompt Zero-shot":
            data = pick_prompt_result(prompt_results, "prompt_bert", 0)
        elif display_name == "BERT Prompt Few-shot":
            data = pick_prompt_result(prompt_results, "prompt_bert", 1)
        elif display_name == "GPT Prompt Zero-shot":
            data = pick_prompt_result(prompt_results, "prompt_gpt", 0)
        elif display_name == "GPT Prompt Few-shot":
            data = pick_prompt_result(prompt_results, "prompt_gpt", 1)
        elif filename is not None:
            data = load_json_if_exists(results_dir / filename)
        else:
            data = None
        if data is not None:
            result_map[display_name] = data
    return result_map


def extract_test_summary(result_payload: dict[str, Any]) -> dict[str, Any] | None:
    metrics = result_payload.get("metrics", {})
    test_metrics = metrics.get("test")
    if not isinstance(test_metrics, dict):
        return None
    required = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    if not all(name in test_metrics for name in required):
        return None
    return test_metrics


def write_comparison_csv(
    output_path: Path,
    result_map: dict[str, dict[str, Any]],
    skip_gpt: bool,
) -> None:
    rows = []
    for display_name, _ in COMPARISON_ORDER:
        if skip_gpt and display_name.startswith("GPT Prompt"):
            rows.append(
                {
                    "model": display_name,
                    "accuracy": "",
                    "macro_precision": "",
                    "macro_recall": "",
                    "macro_f1": "",
                }
            )
            continue

        result_payload = result_map.get(display_name)
        summary = extract_test_summary(result_payload) if result_payload else None
        rows.append(
            {
                "model": display_name,
                "accuracy": "" if summary is None else summary["accuracy"],
                "macro_precision": "" if summary is None else summary["macro_precision"],
                "macro_recall": "" if summary is None else summary["macro_recall"],
                "macro_f1": "" if summary is None else summary["macro_f1"],
            }
        )

    write_csv(
        rows,
        fieldnames=["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"],
        output_path=output_path,
    )


def select_best_model(result_map: dict[str, dict[str, Any]], skip_gpt: bool) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    best_name = None
    best_payload = None
    best_score = None
    for display_name, payload in result_map.items():
        if skip_gpt and display_name.startswith("GPT Prompt"):
            continue
        if payload.get("status") != "success":
            continue
        summary = extract_test_summary(payload)
        if summary is None:
            continue
        score = float(summary["macro_f1"])
        if best_score is None or score > best_score:
            best_name = display_name
            best_payload = payload
            best_score = score
    return best_name, best_payload


def to_chinese_labels(labels: list[str]) -> list[str]:
    return [LABEL_TO_CHINESE.get(label, label) for label in labels]


def build_case_reason(true_label: str, pred_label: str, text: str) -> str:
    tokens = [token for token in tokenize_with_jieba(text) if token.strip()]
    keywords = "、".join(tokens[:5]) if tokens else "文中关键词"
    pair = frozenset({true_label, pred_label})
    if pair == frozenset({"politics", "military"}):
        return f"片段中出现“{keywords}”等局势或行动类词汇，政治和军事语义高度交叉，模型更依赖事件词而不是文章主旨。"
    if pair == frozenset({"economy", "politics"}):
        return f"片段含有“{keywords}”等政策与市场共现词，模型容易把政策性经济报道误归到政治类。"
    if pair == frozenset({"education", "art"}):
        return f"片段中的“{keywords}”更像文化或宣传语境，掩盖了教育主题，模型受表层词面影响较大。"
    if pair == frozenset({"environment", "medicine"}):
        return f"片段中“{keywords}”涉及健康与环境双重线索，模型难以区分公共卫生议题和环境议题。"
    return f"片段中出现“{keywords}”等高频词，与 `{pred_label}` 类训练样本存在词面重叠，但文章主旨更接近 `{true_label}`。"


def write_case_study(output_path: Path, prediction_path: Path, best_model_name: str, macro_f1: float) -> None:
    rows = read_csv_rows(prediction_path)
    error_rows = [row for row in rows if row.get("label") != row.get("pred_label")]
    if not error_rows:
        output_path.write_text(
            f"# Case Study\n\n最佳模型：{best_model_name}\n\n测试集上没有发现错误预测样本。\n",
            encoding="utf-8",
        )
        return

    confusion_counter = Counter((row["label"], row["pred_label"]) for row in error_rows)
    selected_rows = []
    used_pairs = set()
    for (true_label, pred_label), _ in confusion_counter.most_common():
        for row in error_rows:
            pair = (row["label"], row["pred_label"])
            if pair == (true_label, pred_label) and pair not in used_pairs:
                selected_rows.append(row)
                used_pairs.add(pair)
                break
        if len(selected_rows) >= 3:
            break
    if len(selected_rows) < 2:
        for row in error_rows:
            if row not in selected_rows:
                selected_rows.append(row)
            if len(selected_rows) >= 2:
                break

    lines = [
        "# Case Study",
        "",
        f"最佳模型：{best_model_name}",
        f"Test Macro-F1：{macro_f1:.4f}",
        "",
    ]
    for index, row in enumerate(selected_rows[:3], start=1):
        text_snippet = str(row["text"]).replace("\n", " ").strip()
        if len(text_snippet) > 220:
            text_snippet = text_snippet[:220] + "..."
        true_label = str(row["label"])
        pred_label = str(row["pred_label"])
        lines.extend(
            [
                f"## Case {index}",
                "",
                f"- 原始文本片段：{text_snippet}",
                f"- 真实标签：{LABEL_TO_CHINESE.get(true_label, true_label)} ({true_label})",
                f"- 预测标签：{LABEL_TO_CHINESE.get(pred_label, pred_label)} ({pred_label})",
                f"- 错误原因分析：{build_case_reason(true_label, pred_label, str(row['text']))}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    logger = setup_logging("INFO", name="stage3_report")

    result_map = load_result_map(results_dir)
    for display_name, _ in COMPARISON_ORDER:
        if args.skip_gpt and display_name.startswith("GPT Prompt"):
            logger.info("Skipping %s because --skip_gpt is enabled.", display_name)
            continue
        if display_name not in result_map:
            logger.info("No completed result file found for %s. Its comparison row will remain empty.", display_name)
    comparison_path = results_dir / "all_model_comparison.csv"
    write_comparison_csv(comparison_path, result_map, skip_gpt=args.skip_gpt)
    logger.info("Saved model comparison table to %s", comparison_path)

    best_model_name, best_payload = select_best_model(result_map, skip_gpt=args.skip_gpt)
    if best_model_name is None or best_payload is None:
        logger.warning("No successful model result was found for confusion matrix or case study generation.")
        return 0

    best_test_metrics = best_payload["metrics"]["test"]
    matrix = best_test_metrics.get("confusion_matrix")
    labels = best_test_metrics.get("labels", [])
    confusion_path = figures_dir / "confusion_matrix.png"
    save_confusion_matrix_figure(
        matrix=matrix,
        labels=list(labels),
        output_path=confusion_path,
        title=f"Best Model: {best_model_name}",
    )
    logger.info("Saved confusion matrix to %s", confusion_path)

    prediction_path = Path(str(best_payload["prediction_path"]))
    case_study_path = results_dir / "case_study.md"
    write_case_study(
        output_path=case_study_path,
        prediction_path=prediction_path,
        best_model_name=best_model_name,
        macro_f1=float(best_test_metrics["macro_f1"]),
    )
    logger.info("Saved case study to %s", case_study_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
