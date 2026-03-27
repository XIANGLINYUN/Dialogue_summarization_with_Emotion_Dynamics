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


def run_topic_blanc_eval(mode="help"):
    """
    Evaluate topic-level summaries with BLANC.
    - mode="help" → BLANC-Help
    - mode="tune" → BLANC-Tune
    """
    stage1_summary_path = config.SUMMARIZATION_OUTPUT_PATH

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    blanc_help = BlancHelp(device=device)
    blanc_tune = BlancTune(device=device, finetune_mask_evenly=False, show_progress_bar=False)

    overall_score, count = 0.0, 0

    for file in os.listdir(stage1_summary_path):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(stage1_summary_path, file), "r") as f:
            data = json.load(f)

        for topic, content in data.items():
            transcript = content["transcript"]

            if "\n\n" in content["summary"]:
                summary = content["summary"].split("\n\n", 1)[1]
            else:
                summary = content["summary"]

            try:
                if mode == "help":
                    score = blanc_help.eval_once(transcript, summary)
                else:
                    score = blanc_tune.eval_once(transcript, summary)
            except Exception as e:
                print(f"⚠️ BLANC failed for {file}, topic {topic}: {e}")
                continue

            overall_score += float(score)
            print(f"Processed {file}, topic {topic}: BLANC {mode} score = {score:.4f}")
            count += 1

    avg_score = overall_score / count if count > 0 else 0.0
    print(f"\n[Topic] BLANC {mode.capitalize()} Score: {avg_score:.4f}")
    return avg_score


def run_overall_blanc_eval(mode="help"):
    over_summary_path = config.OVERALL_SUMMARIZATION_OUTPUT_PATH

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    blanc_help = BlancHelp(device=device)
    blanc_tune = BlancTune(device=device, finetune_mask_evenly=False, show_progress_bar=False)

    overall_score, count = 0.0, 0

    for file in os.listdir(over_summary_path):
        if not file.endswith(".txt"):
            continue

        file_base = file.replace(".txt", "")
        print(f"Processing summary for {file_base}")

        # load summary
        with open(os.path.join(over_summary_path, file), "r") as f:
            data = f.read()
            summary = data.split(":", 1)[1].strip() if ":" in data else data.strip()
            print("---",summary)

        if config.DATASET == "AMI":
            print("processing file", file)
            with open(os.path.join(config.TRANSCRIPT_FILE_PATH, file), "r") as f:
                lines = f.readlines()
                transcript = ""
                for line in lines:
                    transcript = transcript + line.strip() + " "

        else:
            # find all matching transcript parts
            with open(os.path.join(config.TRANSCRIPT_FILE_PATH, file_base + ".txt"), "r") as f:
                transcript = f.read().strip()
            print(transcript)
            # transcript_parts = sorted(
            #     glob.glob(os.path.join(config.TRANSCRIPT_FILE_PATH, f"{file_base}*.txt"))
            # )

            # if not transcript_parts:
            #     print(f"⚠️ No transcripts found for {file_base}")
            #     continue

            # transcript = ""
            # for part_file in transcript_parts:
            #     with open(part_file, "r") as f:
            #         for line in f:
            #             if not line.strip():
            #                 continue
            #             if "]:" in line:
            #                 utterance = line.split("]:", 1)[1]
            #                 transcript += utterance.strip() + " "

        try:
            if mode == "help":
                print("Using BLANC-Help")
                score = blanc_help.eval_once(transcript, summary)
            else:
                score = blanc_tune.eval_once(transcript, summary)
        except Exception as e:
            print(f"⚠️ BLANC failed for {file}: {e}")
            continue

        overall_score += float(score)
        count += 1

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
        config.PROMPT_VERSION = args.prompt_version
        config.OVERALL_PROMPT_VERSION = args.prompt_version
        config.SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH, "summarization", args.prompt_version
        )
        config.OVERALL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH, "overall_summarization", args.prompt_version
        )

    if args.mode in ("topic", "both"):
        run_topic_blanc_eval(mode=args.blanc_mode)
    if args.mode in ("overall", "both"):
        run_overall_blanc_eval(mode=args.blanc_mode)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safe on macOS
    
    main()
    
