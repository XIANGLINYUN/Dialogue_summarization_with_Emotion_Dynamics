import os
import json
import glob
import argparse
import torch
import multiprocessing as mp
from blanc import BlancHelp, BlancTune

try:
    import config
except ImportError:
    from .. import config


def get_device() -> torch.device:
    """Prefer CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    return device


def build_blanc(mode: str, device: torch.device):
    """Instantiate only the requested BLANC variant."""
    if mode == "help":
        return BlancHelp(device=device)
    elif mode == "tune":
        return BlancTune(
            device=device,
            finetune_mask_evenly=False,
            show_progress_bar=False,
        )
    raise ValueError(f"Unsupported BLANC mode: {mode}")


def load_topic_transcript(transcript_file: str) -> str:
    transcript = []
    with open(transcript_file, "r") as f:
        json_data = json.load(f)
        for _, item in json_data.items():
            utterance = item.get("utterance", "").strip()
            if utterance:
                transcript.append(utterance)
    return " ".join(transcript)


def load_overall_summary(summary_file: str) -> str:
    with open(summary_file, "r") as f:
        data = f.read()
    return data.split(":", 1)[1].strip() if ":" in data else data.strip()


def load_ami_transcript(transcript_file: str) -> str:
    with open(transcript_file, "r") as f:
        return " ".join(line.strip() for line in f if line.strip())


def load_iemocap_transcript(file_base: str) -> str:
    transcript_parts = sorted(
        glob.glob(os.path.join(config.TRANSCRIPT_FILE_PATH, f"{file_base}*.txt"))
    )

    if not transcript_parts:
        return ""

    transcript = []
    for part_file in transcript_parts:
        with open(part_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "]:" in line:
                    utterance = line.split("]:", 1)[1].strip()
                    if utterance:
                        transcript.append(utterance)
                else:
                    transcript.append(line)
    return " ".join(transcript)


def run_topic_blanc_eval(mode="help"):
    """
    Evaluate topic-level summaries with BLANC.
    - mode="help" → BLANC-Help
    - mode="tune" → BLANC-Tune
    """
    stage1_summary_path = config.SUMMARIZATION_OUTPUT_PATH

    device = get_device()
    blanc_model = build_blanc(mode, device)

    overall_score, count = 0.0, 0

    for file in os.listdir(stage1_summary_path):
        summary_file = os.path.join(stage1_summary_path, file)
        transcript_file = os.path.join(config.INDIVIDUAL_EMOTION_TRANSCRIPT_PATH, file)

        if not os.path.isfile(summary_file) or not os.path.isfile(transcript_file):
            continue

        transcript = load_topic_transcript(transcript_file)

        with open(summary_file, "r") as f:
            data = json.load(f)
            summary = data.get("summary", "").strip()

        if not transcript or not summary:
            print(f"⚠️ Skipping {file}: empty transcript or summary")
            continue

        try:
            score = blanc_model.eval_once(transcript, summary)
        except Exception as e:
            print(f"⚠️ BLANC failed for {file}: {e}")
            continue

        overall_score += float(score)
        print(f"Processed {file}: BLANC-{mode.capitalize()} score = {score:.4f}")
        count += 1

    print(f"Total files processed: {count}")
    print(f"Total BLANC {mode} score: {overall_score:.4f}")

    avg_score = overall_score / count if count > 0 else 0.0
    print(f"\n[Topic] BLANC {mode.capitalize()} Score: {avg_score:.4f}")

    return avg_score


def run_overall_blanc_eval(mode="help"):
    over_summary_path = config.OVERALL_SUMMARIZATION_OUTPUT_PATH

    device = get_device()
    blanc_model = build_blanc(mode, device)

    overall_score, count = 0.0, 0

    for file in os.listdir(over_summary_path):
        if not file.endswith(".txt"):
            continue

        file_base = file.replace(".txt", "")
        print(f"Processing summary for {file_base}")

        summary_path = os.path.join(over_summary_path, file)
        summary = load_overall_summary(summary_path)

        if config.DATASET == "AMI":
            transcript_path = os.path.join(config.TRANSCRIPT_FILE_PATH, file)
            if not os.path.isfile(transcript_path):
                print(f"⚠️ No transcript found for {file}")
                continue
            transcript = load_ami_transcript(transcript_path)
        else:
            transcript = load_iemocap_transcript(file_base)
            if not transcript:
                print(f"⚠️ No transcripts found for {file_base}")
                continue

        if not summary:
            print(f"⚠️ Empty summary for {file}")
            continue

        try:
            score = blanc_model.eval_once(transcript, summary)
        except Exception as e:
            print(f"⚠️ BLANC failed for {file}: {e}")
            continue

        overall_score += float(score)
        count += 1
        print(f"Processed {file}: BLANC-{mode.capitalize()} score = {score:.4f}")

    avg_score = overall_score / count if count > 0 else 0.0
    print(f"\n[Overall] BLANC {mode.capitalize()} Score: {avg_score:.4f}")
    return avg_score


def main():
    parser = argparse.ArgumentParser(description="Run BLANC evaluation on summaries")
    parser.add_argument(
        "--blanc_mode",
        choices=["help", "tune"],
        default="help",
        help="Which BLANC variant to run: 'help' (default) or 'tune'."
    )
    parser.add_argument(
        "--mode",
        choices=["topic", "overall", "both"],
        default="both",
        help="Run BLANC on 'topic' summaries, 'overall' summaries, or 'both'."
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.prompt_version:
        for prompt_version in config.SETUP_SET:

            config.PROMPT_VERSION = prompt_version
            config.OVERALL_PROMPT_VERSION = prompt_version
            config.SUMMARIZATION_OUTPUT_PATH = os.path.join(
                config.PROCESSED_OUTPUT_PATH, "individual_summarization", config.MODEL
            )
            config.OVERALL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
                config.PROCESSED_OUTPUT_PATH, "overall_summarization_with_individual", prompt_version
            )

    if args.mode in ("topic", "both"):
        run_topic_blanc_eval(mode=args.blanc_mode)
    if args.mode in ("overall", "both"):
        run_overall_blanc_eval(mode=args.blanc_mode)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()