import os
import json
import requests
import re
import time
import argparse


MAX_RETRIES = 5
FILE_DELAY = 2.0   # delay between processing different files

try:
    from .. import config       # works when run from project_root
except ImportError:
    import config  


URL = "http://127.0.0.1:11434/api/chat"


# ---------------- Utils ---------------- #
def process_list(lst):
    """Post-process raw emotion lists."""
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


def llama3_emotion_classify(prompt, model="llama3.1:8b"):
    """Call LLaMA3 to classify emotions in summaries."""
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in analyzing emotions in conversations. "
                    "You only know the following emotion labels: "
                    "[anger, happiness, excited, sadness, frustration, fear, surprise, other, neutral]. "
                    "You must use these exact labels in your response."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["message"]["content"]


def extract_json(text):
    """Extract valid JSON from LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            candidate = candidate.replace("\n", " ").replace("\r", " ")
            candidate = re.sub(r",\s*}", "}", candidate)
            candidate = re.sub(r",\s*]", "]", candidate)
            return json.loads(candidate)
        raise ValueError("No JSON object found in LLM output")


def safe_emotion_extraction_individual(summary, model="llama3.1:8b", max_retries=MAX_RETRIES, delay=1.5):
    """
    Shared helper to call LLM and extract emotions for either topic or overall summary.
    Retries on invalid JSON with delay between attempts.
    """
    prompt = f"""
You are given a conversation summary. Your task is to:

1. Identify all **emotion expressions** in the summary.  
2. Categorize each expression into one of the following emotions:  
[anger, happiness, excited, sadness, frustration, surprise, other, neutral].  
MAKE SURE TO USE THESE EXACT LABELS. NO OTHER LABELS ARE ALLOWED.  
3. Maintain the **emotion trajectory** (the order in which emotions appear).  

Summary: "{summary}"

Respond strictly in the following JSON format:
{{
"Speaker": ["emotion1", "emotion2", "emotion3"],
}}

Rules:
- Only use speaker ID as key.  
- ONLY use labels from [anger, happiness, excited, sadness, frustration, surprise, other, neutral].  
- DO NOT add explanations or any other text.
- Make sure the JSON is valid.
"""
    for attempt in range(1, max_retries + 1):
        response = llama3_emotion_classify(prompt, model=model)
        try:
            return extract_json(response)
        except Exception as e:
            print(f"[Attempt {attempt}] Invalid JSON: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                prompt += "\n\nRemember: RETURN ONLY VALID JSON, nothing else."
                time.sleep(delay)
            else:
                raise ValueError(f"Failed after {max_retries} retries. Last response:\n{response}")


def safe_emotion_extraction(summary, model="llama3.1:8b", max_retries=MAX_RETRIES, delay=1.5):
    """
    Shared helper to call LLM and extract emotions for either topic or overall summary.
    Retries on invalid JSON with delay between attempts.
    """
    prompt = f"""
You are given a conversation summary. Your task is to:

1. Identify all **emotion expressions** in the summary.  
2. Categorize each expression into one of the following emotions:  
[anger, happiness, excited, sadness, frustration, surprise, other, neutral].  
MAKE SURE TO USE THESE EXACT LABELS. NO OTHER LABELS ARE ALLOWED.  
3. Maintain the **emotion trajectory** (the order in which emotions appear).  
4. Provide the emotions **separately for each speaker**.

Summary: "{summary}"

Respond strictly in the following JSON format:
{{
"Speaker1": ["emotion1", "emotion2", "emotion3"],
"Speaker2": ["emotion1", "emotion2"],
"Speaker3": ["emotion1", "emotion2"],
"Speaker4": ["emotion1", "emotion2", "emotion3"]
}}

Rules:
- Only use speaker IDs as keys.  
- ONLY use labels from [anger, happiness, excited, sadness, frustration, surprise, other, neutral].  
- DO NOT add explanations or any other text.
- Make sure the JSON is valid.
"""
    for attempt in range(1, max_retries + 1):
        response = llama3_emotion_classify(prompt, model=model)
        try:
            return extract_json(response)
        except Exception as e:
            print(f"[Attempt {attempt}] Invalid JSON: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                prompt += "\n\nRemember: RETURN ONLY VALID JSON, nothing else."
                time.sleep(delay)
            else:
                raise ValueError(f"Failed after {max_retries} retries. Last response:\n{response}")


# ---------------- Topic-wise Evaluation ---------------- #
def evaluate_individual_summaries():

    out_path = os.path.join(config.EMOTION_EVAL_RESULT_PATH, config.SETUP_SET[0] +"_individual.json")
    if os.path.exists(out_path):
        print(f"⏩ Skipping topic-wise evaluation, already exists at {out_path}")
        return

    emotion_labels_gt_pred = {}


    for file in os.listdir(config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH):
        print(f"Processing (topic): {file}")


        # load ground-truth transcript
        transcript = ""
        transcript_file = os.path.join(config.INDIVIDUAL_EMOTION_TRANSCRIPT_PATH, file)
        with open(transcript_file, "r") as f:
            data = json.load(f)

        # ground truth
        items = [data[k] for k in sorted(data, key=lambda x: int(x))]

        emotion_list = {"Speaker": []}
        prev_emotion = None

        for item in items:
            emotion = item["emotion"]
            if emotion != prev_emotion:
                emotion_list["Speaker"].append(emotion)
                prev_emotion = emotion

        # prediction
        pred_ind_wise = []
        with open(os.path.join(config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH, file), "r") as f:
            ind_wise_summary = json.load(f)

            pred = safe_emotion_extraction_individual(ind_wise_summary["summary"])
            pred_ind_wise.append(pred)

        emotion_labels_gt_pred[file] = {"ground_truth": emotion_list, "prediction": pred_ind_wise}

        # delay before moving to next file
        time.sleep(FILE_DELAY)


    with open(out_path, "w") as f:
        json.dump(emotion_labels_gt_pred, f, indent=4)
    print(f"\nTopic-wise results saved to {out_path}")


# ---------------- Overall Evaluation ---------------- #
def evaluate_overall_summaries():

    out_path = os.path.join(config.EMOTION_EVAL_RESULT_PATH, config.SETUP_SET[0] +"_overall_individual_only.json")
    if os.path.exists(out_path):
        print(f"⏩ Skipping overall evaluation, already exists at {out_path}")
        return
    emotion_labels_gt_pred = {}

    for file in os.listdir(os.path.join(config.OVERALL_SUMMARIZATION_OUTPUT_PATH_INDIVIDUAL_ONLY, config.SETUP_SET[0])):
        print(f"Processing (overall): {file}")
        if not file.endswith(".txt"):
            continue
        file_name = file.split(".")[0]

        transcript_file = os.path.join(config.PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH, file_name + ".json")
        if not os.path.exists(transcript_file):
            print(f"⚠️ Missing transcript for {file_name}, skipping.")
            continue

        with open(transcript_file, "r") as f:
            data = json.load(f)

        gt_emotions = {}
        for _, item in data.items():
            gt_emotions.setdefault(item["speaker"], []).append(item["emotion"])# item["speaker"][0] for iemocap
        gt_processed = {spk: process_list(lst) for spk, lst in gt_emotions.items()}


        with open(os.path.join(config.OVERALL_SUMMARIZATION_OUTPUT_PATH_WITH_INDIVIDUAL, file), "r") as f:
            summary = f.read().split(":")[-1].strip()

        pred = safe_emotion_extraction(summary)

        emotion_labels_gt_pred[file_name] = {"ground_truth": gt_processed, "prediction": pred}

        # delay before moving to next file
        time.sleep(FILE_DELAY)

    
    with open(out_path, "w") as f:
        json.dump(emotion_labels_gt_pred, f, indent=4)
    print(f"\nOverall results saved to {out_path}")


# ---------------- Main ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization emotion classification")
    parser.add_argument("--mode", choices=["topic", "overall", "both"], default="both",
                        help="Choose evaluation mode: topic-wise, overall, or both.")
    args = parser.parse_args()

    config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
        config.PROCESSED_OUTPUT_PATH, "individual_summarization", config.MODEL
    )
    config.OVERALL_SUMMARIZATION_OUTPUT_PATH_WITH_INDIVIDUAL = os.path.join(
        config.PROCESSED_OUTPUT_PATH, "overall_summarization_individual_only", config.SETUP_SET[0])

    os.makedirs(config.EMOTION_EVAL_RESULT_PATH, exist_ok=True)


    evaluate_overall_summaries()


if __name__ == "__main__":

    main()
