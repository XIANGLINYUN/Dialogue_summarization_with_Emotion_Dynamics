# import os
# import json
# import requests
# import time

# try:
#     from .. import config       # works when run from project_root
# except ImportError:
#     import config  

# import os, sys
# print("RUNNING FILE:", __file__)
# print("PYTHON:", sys.executable)
# print("CWD:", os.getcwd())




# # URL = "http://localhost:11434/api/chat"
# URL = "http://127.0.0.1:11434/api/chat"


# def llama3(sentences, model=config.MODEL, max_retries=5, delay=2.0, save_prompt=False):
#     # prompt_path = os.path.join("/Users/linyunxiang/NL/TUD/Project/2025/dynamic_summary/project/w_o_emotion/topic_prompts",config.PROMPT_VERSION.split("_")[1] + ".txt")

#     user_part = '''
#     I have a transcript of a speaker in JSON format. 

#     Your task:
#     - Summarize the conversation into a shorter narrative that captures both the main points and the speaker's emotional states. 
#     - Clearly integrate "who is doing what with what emotion."
#     - Keep the summary proportional in length (informative but concise). 
#     - Use only the listed emotions (anger, happiness, excited, sadness, frustration, fear, surprise, neutral). 
#     - Output format must be exactly: 
#     Summary: summary content

#     Do not provide explanations, notes, or multiple versions—only the final summary.

#     Transcript:
#     '''
#     user_message = user_part.strip()+ json.dumps(sentences)
#     print("User message to Llama3:", len(user_message))

#     data = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "hi"
#                 # "content": "You are an expert in summarization, and you know how to generate informative summaries with emotion information. You only know the following emotions: anger, happiness, excited, sadness, frustration, fear, surprise, and neutral."
#             },
#             {
#                 "role": "user",
#                 "content": "how are you?"
#             },
#         ],
#         "stream": False,
#         "options": {"num_ctx": 2048},
#     }


#     headers = {"Content-Type": "application/json"}

#     for attempt in range(1, max_retries + 1):
#         try:
#             response = requests.post(URL, headers=headers, json=data)
#             response.raise_for_status()
#             result = response.json()["message"]["content"]
#             if result.strip():  # sanity check
#                 return result
#             else:
#                 raise ValueError("Empty response from model")
#         except Exception as e:
#             print(f"[Attempt {attempt}] Failed to get summary: {e}")
#             if attempt < max_retries:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("❌ All retries failed. Returning empty summary.")
#                 return "Summary: (failed to generate)"


# def summarize_topics():
#     """
#     Perform topic-wise summarization using segmentation + transcript with emotions.
#     """
#     os.makedirs(config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH, exist_ok=True)

#     for file in os.listdir(config.INDIVIDUAL_EMOTION_TRANSCRIPT_PATH):

#         output_file = os.path.join(
#             config.INDIVIDUAL_SUMMARIZATION_OUTPUT_PATH, file.replace(".txt", ".json")
#         )
#         if os.path.exists(output_file):
#             print(f"⏩ Skipping {file}, already summarized at {output_file}")
#             continue

#         print(f"Processing file: {file}")
#         individual_transcript = {}

#         # load topics
#         with open(os.path.join(config.SEGMENTATION_OUTPUT_PATH, file.split("&")[0]+".json"), "r") as f:
#             topics = f.read()
#             if topics == {}:
#                 print(f"⚠️ No topics found in {file}, skipping.")
#                 asd
#             topics = "[" + topics.split("[")[1].split("]")[0].strip() + "]"
#             topics = json.loads(topics)

#             transcript_emotion_file = os.path.join(
#                 config.PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH, file.split("&")[0] + ".json"
#             )
#             transcript_emotion_mapping_file = os.path.join(
#                 config.INDIVIDUAL_TRANSCRIPT_MAPPING_PATH, file.replace(".txt", ".json")
#             )
#             transcript_speaker_file = os.path.join(
#                 config.INDIVIDUAL_EMOTION_TRANSCRIPT_PATH, file)
#         with open(transcript_emotion_file, "r") as f:
#             transcript_json_emotion = json.load(f)

#         with open(transcript_emotion_mapping_file, "r") as f:
#             transcript_mapping = json.load(f)

#         with open(transcript_speaker_file, "r") as f:
#             transcript_speaker_file = json.load(f)
#         # build per-topic transcripts
#         ind = 0
#         for i in range (0,len(transcript_json_emotion)+1):
            
#             if str(i) in transcript_mapping.keys():
#                 # print("----",transcript_mapping[str(i)])
#                 individual_transcript[ind] = transcript_speaker_file[str(transcript_mapping[str(i)])]
#                 ind += 1
#             else:
#                 continue
#         # print(individual_transcript)

#         response = llama3(individual_transcript)   # now with retry + delay

#         individual_summary = {
#             "transcript": individual_transcript,
#             "summary": response,
#         }


#         # save
#         with open(output_file, "w") as f:
#             json.dump(individual_summary, f, indent=4)
#         print(f"✅ Summary saved to {output_file}")


# def main():

#     config.PROMPT_VERSION = "e+t+"
#     summarize_topics()


# if __name__ == "__main__":
#     main()

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

    user_part = """
I have a transcript of a speaker in JSON format.

Your task:
- Summarize the conversation into a shorter narrative that captures both the main points and the speaker's emotional states.
- Clearly integrate "who is doing what with what emotion."
- Keep the summary proportional in length (informative but concise).
- Use only the listed emotions (anger, happiness, excited, sadness, frustration, fear, surprise, neutral).
- Output format must be exactly:
Summary: summary content

Do not provide explanations, notes, or multiple versions—only the final summary.

Transcript:
""".strip()

    user_message = user_part + "\n" + json.dumps(sentences, ensure_ascii=False)
    print("Model:", repr(model))
    print("User message length:", len(user_message))

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in summarization, and you know how to generate informative "
                    "summaries with emotion information. You only know the following emotions: "
                    "anger, happiness, excited, sadness, frustration, surprise, and neutral."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        # Use a sane context window; adjust upward only if needed and stable.
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

        # Load topics (you don't currently use `topics` later, but keep parsing if needed)
        seg_path = os.path.join(config.SEGMENTATION_OUTPUT_PATH, file.split("&")[0] + ".json")
        with open(seg_path, "r") as f:
            topics_raw = f.read()

        # Fix: topics_raw is a string, so compare as string
        if not topics_raw.strip():
            print(f"⚠️ No topics found in {file}, skipping.")
            continue

        # Your original parsing logic (kept as-is, but guarded)
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
                    individual_transcript[ind] = transcript_speaker_json[mapped_idx]
                    ind += 1

        response = llama3(individual_transcript)
        print(response)

        individual_summary = {
            "transcript": individual_transcript,
            "summary": response,
        }

        with open(output_file, "w") as f:
            json.dump(individual_summary, f, indent=4, ensure_ascii=False)

        print(f"✅ Summary saved to {output_file}")


def main():
    config.PROMPT_VERSION = "e+t+"
    summarize_topics()


if __name__ == "__main__":
    main()