from openai import OpenAI
import json
import os

client = OpenAI(api_key="OPENAI_API_KEY")

speaker_path = "/Users/linyunxiang/NL/TUD/Project/2025/dynamic_summary/dataset/IEMOCAP_all/Session_all/individual_summarization/llama3_combined"
topic_path = "/Users/linyunxiang/NL/TUD/Project/2025/dynamic_summary/dataset/IEMOCAP_all/Session_all/summarization/8b_e+t+"
output_path = "/Users/linyunxiang/NL/TUD/Project/2025/dynamic_summary/project/annotation/summary1"


for file in os.listdir(topic_path):
    if not file.endswith(".json"):
        continue

    topic_file = os.path.join(topic_path, file)
    speaker_file = os.path.join(speaker_path, file)
    output_file = os.path.join(output_path, file.replace(".json", ".json"))
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping.")
        continue

    if not os.path.exists(speaker_file):
        print(f"Skipping {file}: matching speaker file not found.")
        continue

    with open(topic_file, "r", encoding="utf-8") as f:
        raw_topics = json.load(f)

    topics = {}
    for topic, content in raw_topics.items():
        topics[topic] = content.get("summary", "").strip()

    with open(speaker_file, "r", encoding="utf-8") as f:
        speakers = json.load(f)

    print(topics)
    print(speakers)

    response = client.responses.create(
        model="gpt-5-mini",  # replace with the actual model name you have access to
        input=[
            {
                "role": "system",
                "content": (
                    "You are an expert in summarization. Generate one overall summary "
                    "of the conversation. Use only these emotions: anger, happiness, "
                    "excited, sadness, frustration, surprise, neutral. "
                    "Output format: Summary: ..."
                )
            },
            {
                "role": "user",
                "content": f"""
You are given:
1) A JSON file containing multiple subtopics and their corresponding summaries. Each summary includes emotion information and is segmented from a full conversation.
2) A JSON file containing speaker-specific summaries describing each speaker’s overall behavior and emotions across the entire conversation.

Task:
Using both the subtopic-level summaries and the speaker-specific summaries, write ONE overall summary that captures the conversation’s full trajectory.

Requirements:
- Write an overall summary capturing the trajectory of the conversation.
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
"""
            }
        ]
    )

    print(f"{file} -> {response.output_text}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": response.output_text}, f, ensure_ascii=False, indent=2)
