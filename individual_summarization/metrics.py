import os
import json
import numpy as np
import random
import argparse
import textdistance
from collections import Counter

try:
    from .. import config
except ImportError:
    import config


# ---------------- Helpers ---------------- #
def process_list(lst):
    processed = []
    for i, item in enumerate(lst):
        if item is None:
            continue
        if isinstance(item, list) and len(item) > 1:
            prev_emotion = processed[-1] if processed else None
            next_emotion = None
            for j in range(i + 1, len(lst)):
                if lst[j] is not None:
                    next_emotion = lst[j]
                    break
            if isinstance(prev_emotion, str) and prev_emotion in item:
                item = [prev_emotion]
            elif isinstance(next_emotion, list):
                for emo in next_emotion:
                    if emo in item:
                        item = [emo]
                        break
            elif isinstance(next_emotion, str) and next_emotion in item:
                item = [next_emotion]
        if isinstance(item, list) and len(item) == 1:
            item = item[0]
        if processed and processed[-1] == item:
            continue
        processed.append(item)
    return processed


def calculate_levenshtein_scores(seq1, seq2):
    dist = textdistance.levenshtein(seq1, seq2)
    return 1 - dist / max(len(seq1), len(seq2)) if max(len(seq1), len(seq2)) > 0 else 1.0


def ngrams(seq, n=2):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def ngram_overlap(seq1, seq2, n=2):
    ng1, ng2 = ngrams(seq1, n), ngrams(seq2, n)
    if not ng2:
        return 0.0
    c1, c2 = Counter(ng1), Counter(ng2)
    return sum((c1 & c2).values()) / len(ng2)


def calculate_multi_ngram_overlap(seq1, seq2, max_n=3):
    scores = []
    for n in range(1, max_n + 1):
        if len(seq1) >= n and len(seq2) >= n:
            scores.append(ngram_overlap(seq1, seq2, n=n))
    return sum(scores) / len(scores) if scores else 0.0


def transitions(seq):
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def elementwise_similarity(seq1, seq2):
    s1, s2 = set(seq1), set(seq2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def jaccard_similarity(seq1, seq2):
    t1, t2 = set(transitions(seq1)), set(transitions(seq2))
    if not t1 or not t2:
        return elementwise_similarity(seq1, seq2)
    return len(t1 & t2) / len(t1 | t2)


def cosine_similarity(seq1, seq2):
    t1, t2 = transitions(seq1), transitions(seq2)
    vocab = list(set(t1) | set(t2))
    if not vocab:
        return elementwise_similarity(seq1, seq2)
    v1 = np.array([t1.count(v) for v in vocab])
    v2 = np.array([t2.count(v) for v in vocab])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return elementwise_similarity(seq1, seq2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ---------------- Evaluation ---------------- #
def evaluate_metrics(gt_seq, pred_seq):
    return {
        "levenshtein": calculate_levenshtein_scores(gt_seq, pred_seq),
        "ngram": calculate_multi_ngram_overlap(gt_seq, pred_seq),
        "jaccard": jaccard_similarity(gt_seq, pred_seq),
        "cosine": cosine_similarity(gt_seq, pred_seq),
    }


def normalize_emotion(label):
    mapping = {
        "happy": "happiness",
        "happiness": "happiness",
        "sad": "sadness",
        "sadness": "sadness",
        "excited": "excited",
        "neutral": "neutral",
        "surprise": "surprise",
        "frustration": "frustration",
        "anger": "anger",
    }

    def normalize_one(x):
        if not isinstance(x, str):
            x = str(x)
        x = x.lower()
        normalized = mapping.get(x, "other")
        return normalized if normalized in config.EMOTION_LIST else "other"

    if isinstance(label, list):
        return [normalize_one(x) for x in label]
    else:
        return normalize_one(label)

def merge_consecutive(seq):
    result = []

    def normalize_item(x):
        if x is None:
            return None
        if isinstance(x, str):
            return x.lower()
        if isinstance(x, list):
            vals = []
            for v in x:
                if v is None:
                    continue
                vals.append(str(v).lower())
            return vals
        return str(x).lower()

    for i, item in enumerate(seq):
        prev_emotion = result[-1] if result else None

        next_item = seq[i + 1] if i + 1 < len(seq) else None
        next_norm = normalize_item(next_item)

        if isinstance(next_norm, list):
            next_emotions = next_norm
        elif next_norm is None:
            next_emotions = []
        else:
            next_emotions = [next_norm]

        current = normalize_item(item)

        if current is None:
            continue

        if isinstance(current, list):
            candidates = current
            if not candidates:
                continue

            if prev_emotion in candidates:
                chosen = prev_emotion
            elif any(x in candidates for x in next_emotions):
                chosen = next(x for x in next_emotions if x in candidates)
            else:
                chosen = candidates[0]
        else:
            chosen = current

        if chosen is None:
            continue

        if not result or result[-1] != chosen:
            result.append(chosen)

    return result


def run_metrics(result_json_path):
    with open(result_json_path, "r") as f:
        data_all = json.load(f)

    aggregated = {"levenshtein": [], "ngram": [], "jaccard": [], "cosine": []}
    count = 0
    for _, data in data_all.items():
        gt, pred = data["ground_truth"], data["prediction"]

        if isinstance(gt, list) and isinstance(pred, list):
            for gt_item, pred_item in zip(gt, pred):
                for speaker in gt_item:
                    gt, pred = gt_item["ground_truth"]["Speaker"], list(pred_item["prediction"][0].values())[0]

                    gt_seq = merge_consecutive(gt)
                    pred_seq = merge_consecutive(pred)

                    gt_seq = [normalize_emotion(s) for s in gt_seq]
                    pred_seq = [normalize_emotion(s) for s in process_list(pred_seq)]

                    # min_len = min(len(gt_seq), len(pred_seq))
                    # gt_seq = gt_seq[:min_len]
                    # pred_seq = pred_seq[:min_len]
                    print(f"GT: {gt}")
                    print(f"Processed GT:   {gt_seq}")

                    # print(f"Processed Pred: {pred_seq}")

                    scores = evaluate_metrics(gt_seq, pred_seq)
                    for k in aggregated:
                        aggregated[k].append(scores[k])

        elif isinstance(gt, dict) and isinstance(pred, dict):

            for speaker in gt:
                if len(gt[speaker]) < 8 or len(gt[speaker]) >10 :
                    print(f"Warning: Speaker {speaker} has too few emotions (GT: {len(gt[speaker])}, Pred: {len(pred.get(speaker, []))}), skipping.")
                    continue
                else:
                    count += 1
                if speaker not in pred:
                    print(f"Speaker {speaker} not found in prediction, skipping.")
                    continue
                
                gt_seq = merge_consecutive(gt[speaker])
                pred_seq = merge_consecutive(pred[speaker])
                print(f"GT: {gt[speaker]}")
                print(f"Pred: {pred[speaker]}")

                gt_seq = [normalize_emotion(s) for s in gt_seq]
                pred_seq = [normalize_emotion(s) for s in process_list(pred_seq)]
                print(f"Processed GT: {gt_seq}")

                scores = evaluate_metrics(gt_seq, pred_seq)
                for k in aggregated:
                    aggregated[k].append(scores[k])
    print(aggregated)

    averages = {k: (sum(v) / len(v) if v else 0.0) for k, v in aggregated.items()}

    return averages


def run_metrics_individual(result_json_path):
    with open(result_json_path, "r") as f:
        data_all = json.load(f)

    aggregated = {"levenshtein": [], "ngram": [], "jaccard": [], "cosine": []}

    for _, data in data_all.items():

        gt, pred = data["ground_truth"]["Speaker"], list(data["prediction"][0].values())[0]

        gt_seq = merge_consecutive(gt)
        pred_seq = merge_consecutive(pred)


        gt_seq = [normalize_emotion(s) for s in gt_seq]
        pred_seq = [normalize_emotion(s) for s in process_list(pred_seq)]


        # min_len = min(len(gt_seq), len(pred_seq))
        # gt_seq = gt_seq[:min_len]
        # pred_seq = pred_seq[:min_len]
        print(f"GT: {gt}")

        print(f"Processed GT: {gt_seq}")
        # print(f"Processed Pred: {pred_seq}")

        scores = evaluate_metrics(gt_seq, pred_seq)
        for k in aggregated:
            aggregated[k].append(scores[k])

    averages = {k: (sum(v) / len(v) if v else 0.0) for k, v in aggregated.items()}

    print("\n=== Averages for Individual Summarization ===")
    for metric, score in averages.items():
        print(f"{metric}: {score:.4f}")
    return averages


# ---------------- Main ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Run emotion evaluation metrics")
    parser.add_argument(
        "--mode",
        choices=["topic", "overall", "both", "individual", "overall_individual_topic"],
        default="both",
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    prompt_versions = [args.prompt_version] if args.prompt_version else config.SETUP_SET

    for prompt_version in prompt_versions:
        prompt_version = "amitopic8b_e+t+"
        config.PROMPT_VERSION = prompt_version
        config.OVERALL_PROMPT_VERSION = prompt_version

        config.SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH,
            "individual_summarization",
            config.MODEL,
        )

        config.OVERALL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH,
            "overall_summarization",
            prompt_version,
        )

        modes = ["overall"] if args.mode == "both" else [args.mode]

        # modes = ["individual", "overall_individual_topic", "overall_individual_only"] if args.mode == "both" else [args.mode]

        for mode in modes:
            suffix = mode
            print(f"\n=== Evaluating {mode} (Prompt Version: {prompt_version}) ===")
            result_path = os.path.join(
                config.EMOTION_EVAL_RESULT_PATH,
                f"{prompt_version}_{suffix}.json",
            )
            print(f"Loading results from: {result_path}")
            if suffix == "individual":
                averages = run_metrics_individual(result_path)
            else:
                averages = run_metrics(result_path)

            print(f"\n=== Overall Averages ({mode}) ===")
            for metric, score in averages.items():
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()