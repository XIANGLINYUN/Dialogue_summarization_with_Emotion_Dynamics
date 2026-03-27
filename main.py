import subprocess
import sys

if __name__ == "__main__":
    # print("Starting task segementation pipeline...")
    
    # subprocess.run([sys.executable, "-m", "task_segmentation.runner"], check=True)

    # subprocess.run([sys.executable, "-m", "summarization.topic_summarization"], check=True)
    # subprocess.run([sys.executable, "-m", "summarization.overall_summarization"], check=True)
    # subprocess.run([sys.executable, "-m", "summarization.emotion_classification"], check=True)
    # subprocess.run([sys.executable, "-m", "summarization.combined_eval"], check=True)

    # subprocess.run([sys.executable, "-m", "task_segmentation.runner"], check=True)

    subprocess.run([sys.executable, "-m", "w_o_emotion_both.topic_summarization"], check=True)
    # subprocess.run([sys.executable, "-m", "w_o_emotion.overall_summarization"], check=True)
    # subprocess.run([sys.executable, "-m", "summarization.emotion_classification"], check=True)
    # subprocess.run([sys.executable, "-m", "summarization.combined_eval"], check=True)
    subprocess.run([sys.executable, "-m", "w_o_emotion_both.individual_summarization"], check=True)


    subprocess.run([sys.executable, "-m", "w_o_emotion_both.overall_summarization_individual_topic"], check=True)
    subprocess.run([sys.executable, "-m", "w_o_emotion_both.emotion_classification"], check=True)
    subprocess.run([sys.executable, "-m", "w_o_emotion_both.combined_eval"], check=True)


