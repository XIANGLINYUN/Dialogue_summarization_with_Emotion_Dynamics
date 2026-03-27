import subprocess
import sys

if __name__ == "__main__":
    print("Starting task segementation pipeline...")

    subprocess.run([sys.executable, "-m", "task_segmentation.runner"], check=True)

    subprocess.run([sys.executable, "-m", "summarization_components.topic_summarization"], check=True)
    subprocess.run([sys.executable, "-m", "summarization_components.individual_summarization"], check=True)


    subprocess.run([sys.executable, "-m", "summarization_components.overall_summarization_individual_topic"], check=True)
    subprocess.run([sys.executable, "-m", "summarization_components.emotion_classification"], check=True)
    subprocess.run([sys.executable, "-m", "summarization_components.combined_eval"], check=True)


