import os
import sys
import json
import time
import requests

try:
    from .. import config  # works when run from project_root
except ImportError:
    import config

print("RUNNING FILE:", __file__)
print("PYTHON:", sys.executable)
print("CWD:", os.getcwd())

URL = "http://127.0.0.1:11434/api/chat"

# Reuse a session, but force Connection: close to avoid keep-alive issues
_SESSION = requests.Session()
_SESSION.trust_env = False  # stable behavior (even if no proxies are set)


def ollama_post(url: str, payload: dict, timeout: int = 180) -> requests.Response:
    """Send a POST to Ollama in the same way that worked in your curl-like test."""
    body = json.dumps(payload, ensure_ascii=False)
    headers = {
        "Content-Type": "application/json",
        "Connection": "close",
        "User-Agent": "curl/8.7.1",
    }
    req = requests.Request("POST", url, data=body, headers=headers).prepare()
    return _SESSION.send(req, timeout=timeout)


def llama3(sentences, model=config.MODEL, max_retries=5, delay=2.0, save_prompt=False):
    model = str(model).strip()

    prompt_path = os.path.join(
        config.TOPIC_PROMPT_PATH,
        config.PROMPT_VERSION.split("_")[1] + ".txt"
    )
    print("Prompt version:", config.PROMPT_VERSION)
    print("Full prompt path:", prompt_path)

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
        system_part, user_part = prompt_template.split("=== USER MESSAGE ===")
        system_message = system_part.replace("=== SYSTEM MESSAGE ===", "").strip()
        user_message = user_part.strip() + "\n" + json.dumps(sentences, ensure_ascii=False)

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

    print("Model:", repr(model))
    print("User message length:", len(user_message))

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "stream": False,
        "options": {"num_ctx": 32768},
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = ollama_post(URL, data, timeout=180)

            if resp.status_code != 200:
                print(f"[Attempt {attempt}] HTTP {resp.status_code}")
                print("Body (first 2000 chars):", resp.text[:2000])

            resp.raise_for_status()

            j = resp.json()
            result = j.get("message", {}).get("content", "").strip()
            if not result:
                raise ValueError(f"Empty response content. Top-level keys: {list(j.keys())}")

            return result

        except Exception as e:
            print(f"[Attempt {attempt}] Failed to get summary: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("❌ All retries failed. Returning empty summary.")
                return "Summary: (failed to generate)"


def summarize_topics():
    """Perform topic-wise summarization using segmentation + transcript with emotions."""
    os.makedirs(config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH, exist_ok=True)

    for file in os.listdir(config.INDIVIDUAL_EMOTION_TRANSCRIPT_PATH):
        output_file = os.path.join(
            config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH, file.replace(".txt", ".json")
        )
        if os.path.exists(output_file):
            print(f"⏩ Skipping {file}, already summarized at {output_file}")
            continue

        print(f"Processing file: {file}")
        individual_transcript = {}

        # Load topics
        seg_path = os.path.join(config.SEGMENTATION_OUTPUT_PATH, file.split("&")[0] + ".json")
        with open(seg_path, "r") as f:
            topics_raw = f.read()

        if not topics_raw.strip():
            print(f"⚠️ No topics found in {file}, skipping.")
            continue

        if "[" in topics_raw and "]" in topics_raw:
            topics_str = "[" + topics_raw.split("[", 1)[1].split("]", 1)[0].strip() + "]"
            try:
                topics = json.loads(topics_str)
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse topics JSON for {file}, skipping.")
                continue
        else:
            topics = []

        transcript_emotion_file = os.path.join(
            config.PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH, file.split("&")[0] + ".json"
        )
        transcript_emotion_mapping_file = os.path.join(
            config.INDIVIDUAL_TRANSCRIPT_MAPPING_PATH, file.replace(".txt", ".json")
        )
        transcript_speaker_path = os.path.join(config.INDIVIDUAL_EMOTION_TRANSCRIPT_PATH, file)
        if not os.path.exists(transcript_emotion_file):
            print(f"⚠️ Transcript emotion file not found for {file}: {transcript_emotion_file}")
            continue
        with open(transcript_emotion_file, "r") as f:
            transcript_json_emotion = json.load(f)

        with open(transcript_emotion_mapping_file, "r") as f:
            transcript_mapping = json.load(f)

        with open(transcript_speaker_path, "r") as f:
            transcript_speaker_json = json.load(f)

        # Build per-topic transcripts
        ind = 0
        for i in range(0, len(transcript_json_emotion) + 1):
            key = str(i)
            if key in transcript_mapping:
                mapped_idx = str(transcript_mapping[key])
                if mapped_idx in transcript_speaker_json:
                    if config.PROMPT_VERSION.split("_")[1] in ["e-t-", "e-t+"]:
                        individual_transcript[ind] = transcript_speaker_json[mapped_idx]["utterance"]
                        ind += 1
                    else:
                        individual_transcript[ind] = transcript_speaker_json[mapped_idx]
                        ind += 1

        response = llama3(individual_transcript)

        individual_summary = {
            "transcript": individual_transcript,
            "summary": response,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(individual_summary, f, indent=4, ensure_ascii=False)

        print(f"✅ Summary saved to {output_file}")


def main():
    for prompt_version in config.SETUP_SET:
        config.PROMPT_VERSION = prompt_version
        config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH, "individual_summarization", config.PROMPT_VERSION
        )
        summarize_topics()


if __name__ == "__main__":
    main()