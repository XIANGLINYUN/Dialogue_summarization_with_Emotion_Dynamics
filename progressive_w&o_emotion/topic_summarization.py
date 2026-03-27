import os
import json
import requests
import time

try:
    from .. import config       # works when run from project_root
except ImportError:
    import config  


URL = "http://localhost:11434/api/chat"


def llama3(sentences, model="llama3:8b", max_retries=5, delay=2.0, save_prompt=True):
    """
    Query LLaMA3 for summarization with emotion-constrained output.
    Retries with delay if request fails or response is invalid.
    """
    system_message = "You are an expert in summarization, and you know how to generate informative summaries with emotion information. You only know the following emotions: anger, happiness, excited, sadness, frustration, fear, surprise, and neutral."
    user_message = f"""
    I have a dialog transcript in JSON format. 

    Your task:
    - Summarize the conversation into a shorter narrative that captures both the main points and the speakers' emotional states. 
    - Clearly integrate "who is doing what with what emotion."
    - Keep the summary proportional in length (informative but concise). 
    - Use only the listed emotions (anger, happiness, excited, sadness, frustration, fear, surprise, neutral). 
    - Output format must be exactly: 
    Summary: summary content

    Do not provide explanations, notes, or multiple versions—only the final summary.

    Transcript:

    {sentences}

    """
    if save_prompt:
        prompt_txt = (
            "=== SYSTEM MESSAGE ===\n"
            + system_message.strip()
            + "\n\n=== USER MESSAGE ===\n"
            + user_message.strip()
        )
    if os.path.exists(config.TOPIC_SUMMARIZATION_PROMPT_PATH):
        print(f"ℹ️ Prompt already exists at {config.TOPIC_SUMMARIZATION_PROMPT_PATH}, skipping save.")
    else:
        os.makedirs(os.path.dirname(config.TOPIC_SUMMARIZATION_PROMPT_PATH), exist_ok=True)
        with open(config.TOPIC_SUMMARIZATION_PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write(prompt_txt)
        print(f"✅ {config.PROMPT_VERSION} saved to {config.TOPIC_SUMMARIZATION_PROMPT_PATH}")

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            },
        ],
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()["message"]["content"]
            if result.strip():  # sanity check
                return result
            else:
                raise ValueError("Empty response from model")
        except Exception as e:
            print(f"[Attempt {attempt}] Failed to get summary: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("❌ All retries failed. Returning empty summary.")
                return "Summary: (failed to generate)"


def summarize_topics():
    """
    Perform topic-wise summarization using segmentation + transcript with emotions.
    """
    os.makedirs(config.SUMMARIZATION_OUTPUT_PATH, exist_ok=True)

    for file in os.listdir(config.SEGMENTATION_OUTPUT_PATH):
        if not file.endswith('.txt'):
            continue

        output_file = os.path.join(
            config.SUMMARIZATION_OUTPUT_PATH, file.replace(".txt", ".json")
        )
        if os.path.exists(output_file):
            print(f"⏩ Skipping {file}, already summarized at {output_file}")
            continue

        print(f"Processing file: {file}")
        topic_wise_summary = {}
        topic_transcript_summary = {}

        # load topics
        with open(os.path.join(config.SEGMENTATION_OUTPUT_PATH, file), "r") as f:
            topics = f.read()
            topics = "[" + topics.split("[")[1].split("]")[0].strip() + "]"
            topics = json.loads(topics)

        # load transcript with emotions
        transcript_emotion_file = os.path.join(
            config.PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH, file.replace(".txt", ".json")
        )
        with open(transcript_emotion_file, "r") as f:
            transcript_json_emotion = json.load(f)

        # build per-topic transcripts
        for topic in topics:
            start_sentence = int(topic["start_sentence"])
            end_sentence = int(topic["end_sentence"])
            topic_content = topic["topic"]

            topic_wise_summary[topic_content] = {}
            ind = 0
            for i in range(start_sentence, end_sentence + 1):
                if str(i) in transcript_json_emotion:
                    topic_wise_summary[topic_content][ind] = transcript_json_emotion[str(i)]
                    ind += 1
                else:
                    print(f"Index {i} not found in transcript JSON for file {file}")

        # summarization
        for topic_content, sentences in topic_wise_summary.items():

            transcript_text = " ".join(item["utterance"] for item in sentences.values())
            response = llama3(sentences)   # now with retry + delay

            topic_transcript_summary[topic_content] = {
                "transcript": transcript_text,
                "summary": response,
            }

        # save
        with open(output_file, "w") as f:
            json.dump(topic_transcript_summary, f, indent=4)
        print(f"✅ Summary saved to {output_file}")


def main():
    summarize_topics()


if __name__ == "__main__":
    main()
