import os

# -------------------------
# General configuration
# -------------------------
MODEL = "MODEL_NAME"

# Optional project root for portability
PROJECT_ROOT = "/path/to/project_root"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
PROJECT_CODE_ROOT = os.path.join(PROJECT_ROOT, "project")

# -------------------------
# Input paths
# -------------------------
TRANSCRIPT_FILE_PATH = os.path.join(
    DATASET_ROOT, "dataset_name", "transcript_txt"
)

# -------------------------
# Preprocessing outputs
# -------------------------
PROCESSED_OUTPUT_PATH = os.path.join(
    DATASET_ROOT, "dataset_name"
)

PROCESSED_TRANSCRIPT_PATH_NO_SPEAKER = os.path.join(
    PROCESSED_OUTPUT_PATH, "transcript_json"
)

PROCESSED_TRANSCRIPT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "transcript_json_speaker"
)

PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "transcript_json_emotion_pred"
)

# -------------------------
# Segmentation outputs
# -------------------------
SEGMENTATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "segmentation_output", MODEL
)

# -------------------------
# Segmentation evaluation
# -------------------------
PREDICTION_FILE_PATH = SEGMENTATION_OUTPUT_PATH

REFERENCE_FILE_PATH = os.path.join(
    DATASET_ROOT, "dataset_name", "reference_topics"
)

TRANSCRIPTION_FILE_PATH = os.path.join(
    DATASET_ROOT, "dataset_name", "topic_ref"
)

OUTPUT_COMPARE_JSON = os.path.join(
    DATASET_ROOT, "integrated_result", "compare_dict_model_prompt.json"
)

# -------------------------
# Summarization configuration
# -------------------------
PROMPT_VERSION = "PROMPT_VARIANT"
OVERALL_PROMPT_VERSION = "PROMPT_VARIANT"

SUMMARIZATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "summarization", PROMPT_VERSION
)

TOPIC_SUMMARIZATION_PROMPT_PATH = os.path.join(
    PROJECT_CODE_ROOT, "prompt_dir", "topic_prompts", PROMPT_VERSION + ".txt"
)

OVERALL_SUMMARIZATION_PROMPT_PATH = os.path.join(
    PROJECT_CODE_ROOT, "prompt_dir", "overall_prompts", OVERALL_PROMPT_VERSION + ".txt"
)

# -------------------------
# Overall summarization outputs
# -------------------------
OVERALL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "overall_summarization", OVERALL_PROMPT_VERSION
)

OVERALL_SUMMARIZATION_OUTPUT_PATH_WITH_INDIVIDUAL = os.path.join(
    PROCESSED_OUTPUT_PATH, "overall_summarization_with_individual"
)

OVERALL_SUMMARIZATION_OUTPUT_PATH_INDIVIDUAL_ONLY = os.path.join(
    PROCESSED_OUTPUT_PATH, "overall_summarization_individual_only"
)

# -------------------------
# Evaluation outputs
# -------------------------
EMOTION_EVAL_RESULT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "emotion_eval_result"
)

FINALLY_METRICS = "data"

EMOTION_LIST = [
    "anger",
    "happiness",
    "excited",
    "sadness",
    "frustration",
    "surprise",
    "neutral",
]

# -------------------------
# Experiment setup
# -------------------------
SETUP_SET = ["experiment_variant"]

DATASET = "DATASET_NAME"

# -------------------------
# Individual summarization
# -------------------------
INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "individual_summarization"
)

INDIVIDUAL_EMOTION_TRANSCRIPT_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "individual_transcript"
)

INDIVIDUAL_TRANSCRIPT_MAPPING_PATH = os.path.join(
    PROCESSED_OUTPUT_PATH, "individual_transcript_mapping"
)

# -------------------------
# Prompt directories
# -------------------------
TOPIC_PROMPT_PATH = os.path.join(
    PROJECT_CODE_ROOT, "prompt_dir", "topic_prompts"
)

OVERALL_PROMPT_PATH = os.path.join(
    PROJECT_CODE_ROOT, "prompt_dir", "overall_prompts"
)