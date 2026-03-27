import os
import re
import subprocess
import pandas as pd
from datetime import datetime

try:
    import config
except ImportError:
    from .. import config


def run_script_and_capture(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if result.returncode != 0:
        print("❌ Subprocess failed:", " ".join(cmd))
        print(result.stdout)  # this WILL show NameError / stack trace
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    return result.stdout



def parse_blanc_output(output):
    """Extract BLANC averages from script output."""
    topic_match = re.search(r"\[Topic\] BLANC \w+ Score:\s*([0-9.]+)", output)
    overall_match = re.search(r"\[Overall\] BLANC \w+ Score:\s*([0-9.]+)", output)
    return {
        "blanc_topic": float(topic_match.group(1)) if topic_match else 0.0,
        "blanc_overall": float(overall_match.group(1)) if overall_match else 0.0,
    }


def parse_emotion_output(output):
    """Extract emotion metric averages from script output."""
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

            config.OVERALL_PROMPT_VERSION = prompt_version

            config.OVERALL_SUMMARIZATION_OUTPUT_PATH_WITH_INDIVIDUAL = os.path.join(
                config.PROCESSED_OUTPUT_PATH, "overall_summarization_individual_only", prompt_version
            )
            config.SUMMARIZATION_OUTPUT_PATH = os.path.join(
                config.PROCESSED_OUTPUT_PATH, "individual_summarization", config.MODEL
            )

            # blanc_script = "individual_summarization.blanc"
            emotion_script = "individual_summarization.metrics"

            # --- Run BLANC evaluation (capture stdout -> parse to dict) ---
            # blanc_out = run_script_and_capture([
            #     "python3.10", "-m", blanc_script,
            #     "--mode", "overall",
            #     "--prompt_version", prompt_version
            # ])
            # blanc_results = parse_blanc_output(blanc_out)

            # --- Run Emotion evaluation (capture stdout -> parse to dict) ---
            emotion_out = run_script_and_capture([
                "python3.10", "-m", emotion_script,
                "--mode", "overall",
                "--prompt_version", prompt_version
            ])
            emotion_results = parse_emotion_output(emotion_out)

            # --- Safety: ensure mappings for ** unpack ---
            if not isinstance(blanc_results, dict):
                blanc_results = {"blanc_topic": 0.0, "blanc_overall": 0.0}
            if not isinstance(emotion_results, dict):
                emotion_results = {}

            # Combine results (KEEPING your original Excel format/keys)
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_version_topic": config.PROMPT_VERSION,
                "prompt_version_overall": config.OVERALL_PROMPT_VERSION,
                "session": config.SESSION,
                # **blanc_results,
                **emotion_results,
            }

            # Save to Excel (append mode) - unchanged format
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
