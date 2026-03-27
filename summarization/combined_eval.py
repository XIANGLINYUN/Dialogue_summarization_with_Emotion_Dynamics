import os
import re
import sys
import subprocess
import pandas as pd
from datetime import datetime

try:
    import config
except ImportError:
    from .. import config


def run_script_and_capture(cmd):
    # Force unbuffered Python output so prints appear immediately
    if cmd and os.path.basename(cmd[0]).startswith("python"):
        cmd = cmd[:1] + ["-u"] + cmd[1:]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []

    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
            output="".join(output_lines)
        )

    return "".join(output_lines)


def parse_blanc_output(output):
    topic_match = re.search(r"\[Topic\] BLANC \w+ Score:\s*([0-9.]+)", output)
    overall_match = re.search(r"\[Overall\] BLANC \w+ Score:\s*([0-9.]+)", output)
    return {
        "blanc_topic": float(topic_match.group(1)) if topic_match else 0.0,
        "blanc_overall": float(overall_match.group(1)) if overall_match else 0.0,
    }


def parse_emotion_output(output):
    metrics = {}
    for metric in ["levenshtein", "ngram", "jaccard", "cosine"]:
        matches = re.findall(rf"{metric}:\s*([0-9.]+)", output, re.IGNORECASE)
        if matches:
            metrics.update({
                f"emotion_topic_{metric}": float(matches[0]),
                f"emotion_overall_{metric}": float(matches[-1]),
            })
    return metrics


def main():
    for prompt_version in config.SETUP_SET:
        config.PROMPT_VERSION = prompt_version
        config.OVERALL_PROMPT_VERSION = prompt_version

        config.OVERALL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH, "overall_summarization", config.OVERALL_PROMPT_VERSION
        )
        config.SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH, "summarization", config.PROMPT_VERSION
        )

        # blanc_script = "summarization.blanc"
        emotion_script = "summarization.metrics"

        # blanc_out = run_script_and_capture([
        #     sys.executable, "-m", blanc_script,
        #     "--mode", "both",
        #     "--prompt_version", prompt_version,
        # ])
        # blanc_results = parse_blanc_output(blanc_out)

        emotion_out = run_script_and_capture([
            sys.executable, "-m", emotion_script,
            "--mode", "both",
            "--prompt_version", prompt_version
        ])
        emotion_results = parse_emotion_output(emotion_out)

        if not isinstance(blanc_results, dict):
            blanc_results = {"blanc_topic": 0.0, "blanc_overall": 0.0}
        if not isinstance(emotion_results, dict):
            emotion_results = {}

        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_version_topic": config.PROMPT_VERSION,
            "prompt_version_overall": config.OVERALL_PROMPT_VERSION,
            "session": config.SESSION,
            **blanc_results,
            **emotion_results,
        }

        excel_path = os.path.join(config.FINALLY_METRICS, "evaluation_results.xlsx")
        df_new = pd.DataFrame([results])

        if os.path.exists(excel_path):
            df_existing = pd.read_excel(excel_path)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_excel(excel_path, index=False)
        print(f"\n✅ Results appended to {excel_path}")


if __name__ == "__main__":
    main()