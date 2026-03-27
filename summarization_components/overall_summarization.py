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
_SESSION.headers.update(
    {
        "Content-Type": "application/json",
        "Connection": "close",
        "User-Agent": "curl/8.7.1",
    }
)


def ollama_post(url: str, payload: dict, timeout: int = 180) -> requests.Response:
    """Send a POST to Ollama using requests.post(json=payload)."""
    return _SESSION.post(url, json=payload, timeout=timeout)


def llama3_overall(topics, speakers, model=config.MODEL, max_retries=5, delay=2.0, save_prompt=False):
    model = str(model).strip()

    user_part = f"""
You are given:
1) A JSON file containing multiple subtopics and their corresponding summaries. Each summary includes emotion information and is segmented from a full conversation.
2) A JSON file containing speaker-specific summaries describing each speaker’s overall behavior and emotions across the entire conversation.

Task:
Using both the subtopic-level summaries and the speaker-specific summaries, write ONE overall summary that captures the conversation’s full trajectory.

Requirements:
- Write an overall summary capturing the trajectory of the conversation.
- keep the key events, actions, and other important information of the conversation.
- Integrate the speakers' emotion trajectory.
- Follow the chronological timeline of the dialogue.
- Clearly indicate which speaker performs which action and with what emotion when describing the events in the conversation.
- Keep the summary informative and concise.
- Restrict all emotion-related expressions in the summary to the following set: anger, happiness, excited, sadness, frustration, surprise, neutral.

Return only plain text.

Final output format:
Summary: <summary for the entire conversation>

Input:
Subtopic summaries:
{json.dumps(topics, ensure_ascii=False, indent=2)}

Speaker-specific summaries:
{json.dumps(speakers, ensure_ascii=False, indent=2)}
""".strip()

    if save_prompt:
        prompt_path = os.path.join(os.getcwd(), "overall_prompt_debug.txt")
        with open(prompt_path, "w", encoding="utf-8") as pf:
            pf.write(user_part)
        print(f"Prompt saved to {prompt_path}")

    print("Model:", repr(model))
    print("User message length:", len(user_part))

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in summarization, and you know how to generate "
                    "informative summaries with emotion information. You only know the "
                    "following emotions: anger, happiness, excited, sadness, frustration, "
                    "surprise, and neutral."
                ),
            },
            {"role": "user", "content": user_part},
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


def summarize_overall(output_path=None):
    """Generate overall summarizations from topic-wise summaries."""
    os.makedirs(config.OVERALL_SUMMARIZATION_OUTPUT_PATH_WITH_INDIVIDUAL, exist_ok=True)

    for file in os.listdir(config.SUMMARIZATION_OUTPUT_PATH):
        if not file.endswith(".json"):
            continue
        print(f"Processing file: {file}")
        
        output_txt = os.path.join(output_path, file.replace(".json", ".txt"))

        if os.path.exists(output_txt):
            print(
                f"Overall summarization output path "
                f"{config.OVERALL_SUMMARIZATION_OUTPUT_PATH_WITH_INDIVIDUAL} already exists. "
                f"Skipping overall summarization."
            )
            continue

        print(f"Processing file: {file}")

        with open(os.path.join(config.SUMMARIZATION_OUTPUT_PATH, file), "r", encoding="utf-8") as f:
            topic_transcript_summary = json.load(f)
            topic_summaries = {
                topic: content["summary"]
                for topic, content in topic_transcript_summary.items()
            }

        speaker_summaries = {}
        base_name = file.split(".")[0]

        for speaker_sum in os.listdir(config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH):
            if speaker_sum.startswith(base_name):
                speaker_file = os.path.join(
                    config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH,
                    speaker_sum,
                )

                with open(speaker_file, "r", encoding="utf-8") as f:
                    speaker_id = speaker_sum.split("&")[1].split(".json")[0]
                    speaker_summaries[speaker_id] = json.load(f)["summary"]

        sorted_speaker_summaries = dict(
            sorted(
                speaker_summaries.items(),
                key=lambda x: int(x[0].replace("Speaker", "")),
            )
        )

        response = llama3_overall(topic_summaries, sorted_speaker_summaries)
        

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(response)

        print(f"Overall summary saved to {output_txt}")


def main():
    for prompt_version in config.SETUP_SET:
        config.SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH,
            "summarization",
            prompt_version,
        )
        config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH = os.path.join(
            config.PROCESSED_OUTPUT_PATH,
            "individual_summarization",
            prompt_version,
        )
        output_path = os.path.join(
            config.OVERALL_SUMMARIZATION_OUTPUT_PATH_WITH_INDIVIDUAL, prompt_version
        )
        os.makedirs(output_path, exist_ok=True)
        summarize_overall(output_path=output_path)


if __name__ == "__main__":
    main()