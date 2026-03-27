import os

'''
run inside the project folder
task_segmentation/
│── preprocess.py
│── segmentation.py
│── evaluation.py

python -m task_segmentation.preprocess
python -m task_segmentation.segmentation
python -m task_segmentation.evaluation
'''


# Change session here
SESSION = "Session_all"

# Input paths
# TRANSCRIPT_FILE_PATH = f"../dataset/IEMOCAP_full_release/{SESSION}/dialog/transcriptions"
TRANSCRIPT_FILE_PATH = f"../dataset/IEMOCAP_all/{SESSION}/transcriptions"
EMOTION_FILE_PATH = f"../dataset/IEMOCAP_all/{SESSION}/transcriptions"

# Output base path
PROCESSED_OUTPUT_PATH = f"../dataset/IEMOCAP_all/{SESSION}"
PROCESSED_TRANSCRIPT_PATH_NO_SPEAKER = os.path.join(PROCESSED_OUTPUT_PATH, "transcript_json")
PROCESSED_TRANSCRIPT_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "transcript_json_speaker")
# PROCESSED_TRANSCRIPT_EMOTION_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "transcript_json_emotion")
PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "transcript_json_filtered_emotion")

# Segmentation output
SEGMENTATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "IEMOCAP_llama3_8b_0shot/prompt_v2"
)

# Task segmentation evaluation paths
PREDICTION_FILE_PATH = "../dataset/llama_result/llama3_8b_json"
REFERENCE_FILE_PATH = "../dataset/ami_public_manual_1.6.2/topics"
TRANSCRIPTION_FILE_PATH = "../da taset/topic_ref"
OUTPUT_COMPARE_JSON = "../dataset/integrated_result/compare_dict_llama3_3b_0shot.json"


# Summarization config
#Topic summarization config-

PROMPT_VERSION = "e-t-"
OVERALL_PROMPT_VERSION = "e-t-"
SUMMARIZATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "summarization", PROMPT_VERSION
)
# TOPIC_SUMMARIZATION_PROMPT_PATH = os.path.join(
#     PROCESSED_OUTPUT_PATH, "summarization", "prmopt_version",PROMPT_VERSION+".txt"
# )
# OVERALL_SUMMARIZATION_PROMPT_PATH = os.path.join(
#     PROCESSED_OUTPUT_PATH, "overall_summarization", "prmopt_version", OVERALL_PROMPT_VERSION + ".txt"
# )

TOPIC_SUMMARIZATION_PROMPT_PATH = os.path.join("/Users/linyunxiang/NL/TUD/Project/2025/dynamic_summary/project/w_o_emotion/topic_prompts", PROMPT_VERSION + ".txt")
OVERALL_SUMMARIZATION_PROMPT_PATH = os.path.join("/Users/linyunxiang/NL/TUD/Project/2025/dynamic_summary/project/w_o_emotion/overall_prompts", OVERALL_PROMPT_VERSION + ".txt")

# Overall summarization
OVERALL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "overall_summarization", OVERALL_PROMPT_VERSION
)

# Summarization evaluation
EMOTION_EVAL_RESULT_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "emotion_eval_result")

#Metrics file
FINALLY_METRICS= "data"


EMOTION_LIST = ['anger', 'happiness', 'excited', 'sadness', 'frustration', 'fear', 'surprise', 'other', 'neutral']

MODEL="llama3.1:8b"

SETUP_SET = ["iemocap8b_e-t-", "iemocap70b_e+t-", "iemocap70b_e-t+", "iemocap70b_e+t+"]
# SETUP_SET = ["e-t-", "e+t-", "e-t+", "e+t+"]

DATASET = "iemocap"  # "iemocap" or "ami"