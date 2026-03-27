import os
import json
import re
import time
import requests
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_fast=True
)


try:
    from .. import config  # works when run from project_root
except ImportError:
    import config


URL = "http://localhost:11434/api/chat"
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def llama3(prompt, model=config.MODEL): #or llama3:8b, llama3.2
    """Send prompt to LLaMA3 API."""
    data = {
        "model": model,
        "messages": [{"role": "system", "content": "You are an expert in topic segmentation and must output VALID JSON only."}, 
                     {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": 32768},
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["message"]["content"]


# def build_prompt(transcript_json):
#     return f"""
# # You are given a transcript as a JSON object:
# # - Keys: sentence indices (integers; may appear as strings in JSON)
# # - Values: sentence text (strings)

# # TASK:
# # Segment the transcript into contiguous topic blocks.

# # OUTPUT (STRICT):
# # Return ONLY a JSON array. No prose. No markdown. No code fences.

# # Each array element MUST be an object with EXACTLY these keys:
# # - "topic": string (a short label derived from the actual transcript content)
# # - "start_sentence": integer
# # - "end_sentence": integer

# # HARD CONSTRAINTS:
# # 1) Output ONLY the final JSON array for THIS transcript.
# # 2) DO NOT output the example below. It is only to demonstrate the REQUIRED JSON SHAPE.
# # 3) Topics MUST be derived from THIS transcript (use transcript-specific wording).
# # 4) Blocks MUST be contiguous, non-overlapping, and cover ALL provided indices:
# #    - First block starts at 0
# #    - Each next block starts at (previous end_sentence + 1)
# #    - Final block ends at the maximum sentence index present in the input
# # 5) Use ONLY indices that exist in the input transcript (do not invent missing indices).
# # 6) Prefer fewer, larger coherent blocks. Avoid blocks with fewer than 5 sentences unless unavoidable.

# # EXAMPLE (JSON SHAPE ONLY — DO NOT OUTPUT THIS):
# # [
# #   {{
# #     "topic": "Project timeline planning and milestone alignment",
# #     "start_sentence": 0,
# #     "end_sentence": <intA>
# #   }},
# #   {{
# #     "topic": "Budget allocation and resource constraints",
# #     "start_sentence": <intA>+1,
# #     "end_sentence": max(json index)
# #   }}
# # ]

# # FINAL CHECK BEFORE YOU RESPOND:
# # - Response is valid JSON.
# # - Response is a JSON array (not an object).
# # - All start_sentence and end_sentence values are integers.
# # - No gaps/overlaps; coverage starts at 0 and ends at max provided index.
# # - No forbidden substrings appear.

# # TRANSCRIPT JSON:
# # {json.dumps(transcript_json, ensure_ascii=False, separators=(",", ":"))}
# # """.strip()

def build_prompt(transcript_json):
    """Build segmentation prompt with strict JSON output request."""
    return (f"""Input is a transcript in JSON:
            - keys = sentence indices
            - values = sentence text

            Task:
            Segment the transcript into topic blocks. Prefer fewer, coherent topics, avoid topics with under 5 sentences.

            Rules:
            1) Output ONLY a JSON array.
            2) Each array element MUST be an object with exactly these keys:
                - topic
                - start_sentence
                - end_sentence
            3) start_sentence and end_sentence MUST be integers.
            4) Blocks MUST be contiguous, non-overlapping, and cover all sentences:
            - First block starts with 0
            - Each next block starts at (previous end_sentence + 1).
            - Final block ends at the maximum sentence index in the input.
            EXAMPLE SHAPE (DO NOT COPY THE INDEXES)
            [
                {{"topic": Topic, "start_sentence": 0, "end_sentence": 20}},
                {{"topic": Topic, "start_sentence": 21, "end_sentence": 42}},
                ...
            ]

            DO NOT copy the index from the example
            Return ONLY the JSON array.

            Transcript:
            {json.dumps(transcript_json, ensure_ascii=False)}
            """

            )

# def build_prompt(transcript_json):
#     transcript_str = json.dumps(transcript_json, ensure_ascii=False)
#     return f"""
# I have a meeting transcript that I need segmented into distinct topics. The transcript is provided in JSON format, where each JSON index corresponds to a sentence.

# Please perform detailed topic segmentation on the following transcript.

# Transcript:
# {transcript_str}

# Return only the result in JSON format as:

# [
#   {{
#     "topic": "topic content",
#     "start_sentence": 0,
#     "end_sentence": 20
#   }},
#   {{
#     "topic": "topic content",
#     "start_sentence": 21,
#     "end_sentence": 42
#   }},
#   ...
# ]

# Use the JSON integer index of the sentences from the transcript for "start_sentence" and "end_sentence". Do not include any extra text or explanation—only return the JSON.
# """

            
def extract_json(text):
    """Try to parse JSON, clean up common LLM mistakes."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {text[:200]}")
        candidate = match.group(0)
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        return json.loads(candidate)


def safe_llm_call(prompt):
    """Retry LLM call until valid JSON is returned."""
    for attempt in range(1, MAX_RETRIES + 1):
        response = llama3(prompt)
        try:
            parsed = extract_json(response)
            return parsed, response  # return both parsed and raw
        except Exception as e:
            print(f"[Attempt {attempt}] Failed to parse JSON: {e}")
            if attempt < MAX_RETRIES:
                prompt += "\n\nREMEMBER: Return ONLY valid JSON, no explanations."
                time.sleep(RETRY_DELAY)
            else:
                # raise RuntimeError(
                #     f"Failed after {MAX_RETRIES} retries. Last response:\n{response}"
                # )
                continue


def segment_transcripts():
    os.makedirs(config.SEGMENTATION_OUTPUT_PATH, exist_ok=True)

    for file in os.listdir(config.PROCESSED_TRANSCRIPT_PATH_NO_SPEAKER):
        if not file.endswith(".json"):
            continue

        # check if already processed
        output_json = os.path.join(
            config.SEGMENTATION_OUTPUT_PATH, f"{file.replace('.json', '.json')}"
        )
        if os.path.exists(output_json):
            print(f"⏩ Skipping {file}, already segmented at {output_json}")
            continue

        with open(os.path.join(config.PROCESSED_TRANSCRIPT_PATH_NO_SPEAKER, file), "r") as f:
            transcript_json = json.load(f)

        prompt = build_prompt(transcript_json)

        parsed, raw = safe_llm_call(prompt)
        try:
            # Save raw response (debugging)
            raw_file = os.path.join(
                config.SEGMENTATION_OUTPUT_PATH, f"{file.replace('.json', '.txt')}"
            )
            with open(raw_file, "w") as f:
                f.write(raw)

            # Save clean JSON
            with open(output_json, "w") as f:
                json.dump(parsed, f, indent=4)

            print(f"✅ Segmentation saved: {output_json}")
        except Exception as e:
            print(f"❌ Failed to save segmentation for {file}: {e}")
            continue

def count_tokens_llama(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def average_prompt_tokens():
    token_counts = []
    # for file in ["ES2003a.json", "ES2013a.json","IS1001d.json", "IS1003a.json","IS1004a.json","IS1005a.json", "IS1005b.json","IS1007a.json","IS1007c.json","IS1008c.json","IS1009a.json"]:


    for file in os.listdir(config.PROCESSED_TRANSCRIPT_PATH_NO_SPEAKER):
        if not file.endswith(".json"):
            continue

        with open(
            os.path.join(config.PROCESSED_TRANSCRIPT_PATH_NO_SPEAKER, file),
            "r"
        ) as f:
            transcript_json = json.load(f)

        prompt = build_prompt(transcript_json)
        tokens = count_tokens_llama(prompt)
        print(f"{file}: {tokens} tokens")
        token_counts.append(tokens)

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    print(f"Processed {len(token_counts)} files.")
    return avg_tokens, token_counts

def main():
    avg, all_counts = average_prompt_tokens()
    print(f"Average prompt tokens: {avg:.2f}")
    print(f"Min: {min(all_counts)}, Max: {max(all_counts)}")
    segment_transcripts()

if __name__ == "__main__":
    main()
