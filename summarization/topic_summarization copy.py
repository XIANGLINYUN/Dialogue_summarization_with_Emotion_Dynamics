import os
import json
import requests
import time

try:
    from .. import config       # works when run from project_root
except ImportError:
    import config  


URL = "http://localhost:11434/api/chat"


def llama3(sentences, model="llama3:8b", max_retries=5, delay=2.0):
    """
    Query LLaMA3 for summarization with emotion-constrained output.
    Retries with delay if request fails or response is invalid.
    """
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in summarization, and you know how to generate informative summaries with emotion information. "
                    "You only know the following emotions: anger, happiness, excited, sadness, frustration, fear, surprise, and neutral."
                ),
            },
            {
                "role": "user",
                "content": f"""
I have a dialog transcript that I need to summarize with emotion information. 
The transcript is provided in JSON format, each JSON index corresponds to a sentence, with speaker label, and the emotions that are related to the same sentence. 
Please perform summarization on the following transcript:

{sentences}

Write a short summary capturing the trajectory of a conversation. 
Summarize the context-level information, also focus on how the speakers' emotions integrate into the trajectory summary. 
The summary length should be proportionally shorter than the text length. Make the summary informative and concise.

Restrict all the emotion-related expressions in the summary within the following ones: 
anger, happiness, excited, sadness, frustration, fear, surprise, and neutral.

Return only the result with plain text.  
Summary: summary content

Please make sure that you make only one summarization for the transcript.  
Do not include any extra text or explanation—only return plain text.
"""
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
            config.PROCESSED_TRANSCRIPT_EMOTION_PATH, file.replace(".txt", ".json")
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
