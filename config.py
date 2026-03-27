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
MODEL="llama3.1:8b"
# Input paths
# TRANSCRIPT_FILE_PATH = f"../dataset/IEMOCAP_full_release/{SESSION}/dialog/transcriptions"
TRANSCRIPT_FILE_PATH = "/home/linyun/dynamic_summary/dataset/IEMOCAP_all/Session_all/transcript_txt"

# Output base path
PROCESSED_OUTPUT_PATH = f"../dataset/IEMOCAP_all/Session_all/"
PROCESSED_TRANSCRIPT_PATH_NO_SPEAKER = "/home/linyun/dynamic_summary/dataset/IEMOCAP_all/Session_all/transcript_json"
PROCESSED_TRANSCRIPT_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "transcript_json_speaker")
# PROCESSED_TRANSCRIPT_EMOTION_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "transcript_json_emotion")

# PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "transcript_json_filtered_emotion")
PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "transcript_json_emotion_pred")


# Segmentation output

SEGMENTATION_OUTPUT_PATH = f"/home/linyun/dynamic_summary/dataset/IEMOCAP_all/Session_all/segmentation_output/llama3.1:8b"
# SEGMENTATION_OUTPUT_PATH = "/home/linyun/dynamic_summary/datasetamisegmentation_outpt"

# Task segmentation evaluation paths
# PREDICTION_FILE_PATH = "../dataset/llama_result/llama3_8b_json"
PREDICTION_FILE_PATH = SEGMENTATION_OUTPUT_PATH
REFERENCE_FILE_PATH = "/home/linyun/dynamic_summary/dataset/IEMOCAP_all/Session_all/ami_public_manual_1.6.2/topics"
TRANSCRIPTION_FILE_PATH = "/home/linyun/dynamic_summary/dataset/topic_ref"
OUTPUT_COMPARE_JSON = "/home/linyun/dynamic_summary/dataset/integrated_result/compare_dict_llama3.1_8b_0shot.json"


# Summarization config
#Topic summarization config-

PROMPT_VERSION = "e+t+"
OVERALL_PROMPT_VERSION = "e+t+"
SUMMARIZATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "summarization", PROMPT_VERSION
)
# TOPIC_SUMMARIZATION_PROMPT_PATH = os.path.join(
#     PROCESSED_OUTPUT_PATH, "summarization", "prmopt_version",PROMPT_VERSION+".txt"
# )
# OVERALL_SUMMARIZATION_PROMPT_PATH = os.path.join(
#     PROCESSED_OUTPUT_PATH, "overall_summarization", "prmopt_version", OVERALL_PROMPT_VERSION + ".txt"
# )

TOPIC_SUMMARIZATION_PROMPT_PATH = os.path.join("/home/linyun/dynamic_summary/project/w_o_emotion/topic_prompts", PROMPT_VERSION + ".txt")
OVERALL_SUMMARIZATION_PROMPT_PATH = os.path.join("/home/linyun/dynamic_summary/project/w_o_emotion/overall_prompts", OVERALL_PROMPT_VERSION + ".txt")

# Overall summarization
OVERALL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "overall_summarization", OVERALL_PROMPT_VERSION
)

OVERALL_SUMMARIZATION_OUTPUT_PATH_WITH_INDIVIDUAL= os.path.join(
    PROCESSED_OUTPUT_PATH, "overall_summarization_with_individual",
)

OVERALL_SUMMARIZATION_OUTPUT_PATH_INDIVIDUAL_ONLY = os.path.join(
    PROCESSED_OUTPUT_PATH, "overall_summarization_individual_only"
)
# Summarization evaluations
EMOTION_EVAL_RESULT_PATH = os.path.join(PROCESSED_OUTPUT_PATH, "emotion_eval_result")

#Metrics file
FINALLY_METRICS= "data"


EMOTION_LIST = ['anger', 'happiness', 'excited', 'sadness', 'frustration', 'surprise', 'neutral']


# SETUP_SET = ["iemo8b_e-t-", "iemo8b_e+t-", "iemo8b_e-t+", "iemo8b_e+t+"]
SETUP_SET = ["newiemopic3b_e+t+"]
             #,"amitopic8b_e-t-","amitopic8b_e+t-","amitopic8b_e-t+"]
# SETUP_SET = ["e-t-", "e+t-", ,"e-t+", "e+t+"]


DATASET = "AMI"
INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "individual_summarization"
)

INDIVIDUAL_EMOTION_TRANSCRIPT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "individual_transcript"
)

INDIVIDUAL_TRANSCRIPT_MAPPING_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "individual_transcript_mapping"
)

TOPIC_PROMPT_PATH = "/home/linyun/dynamic_summary/project/w_o_emotion/topic_prompts"
OVERALL_PROMPT_PATH = "/home/linyun/dynamic_summary/project/w_o_emotion/overall_prompts"

