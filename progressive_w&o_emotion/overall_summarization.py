import os
import json
import requests
try:
    from .. import config       # works when run from project_root
except ImportError:
    import config  

URL = "http://localhost:11434/api/chat"


def llama3_overall(sub_topic_summaries, model="llama3:8b", save_prompt=True):
    """
    Query LLaMA3 to generate an overall summary given sub-topic summaries.
    Optionally saves the prompt used for reproducibility/versioning.
    """
    system_message = (    
        "You are an expert in summarization, and you know how to generate informative summaries."
    )

    user_message = f'''
I have a JSON file containing several topics and corresponding summaries with emotion information, which are segmented from an entire conversation.
Please perform summarization on the following sub topics and their summaries:

{sub_topic_summaries}

Write an overall summary capturing the trajectory of the conversation. 
Summarize the context-level information, also integrate the speakers' emotions trajectory, keep the speaker information. Make the summary informative and concise.

Restrict all the emotion-related expressions in the summary within the following ones: 
anger, happiness, excited, sadness, frustration, fear, surprise, and neutral.

Return only the result with plain text. Final output format: 
Summary: #summary for the entire conversation#

Please make sure that you make only one summarization for the given sub topic summaries.  
Do not include any extra text or explanation—only return plain text.
'''
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


    if save_prompt:
        prompt_txt = (
            "=== SYSTEM MESSAGE ===\n"
            + system_message.strip()
            + "\n\n=== USER MESSAGE ===\n"
            + user_message.strip()
        )
        if os.path.exists(config.OVERALL_SUMMARIZATION_PROMPT_PATH):
            print(f"ℹ️ Prompt already exists at {config.OVERALL_SUMMARIZATION_PROMPT_PATH}, skipping save.")
        else:
            os.makedirs(os.path.dirname(config.OVERALL_SUMMARIZATION_PROMPT_PATH), exist_ok=True)
            with open(config.OVERALL_SUMMARIZATION_PROMPT_PATH, "w", encoding="utf-8") as f:
                f.write(prompt_txt)
            print(f"✅ {config.OVERALL_PROMPT_VERSION} saved to {config.OVERALL_SUMMARIZATION_PROMPT_PATH}")

    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["message"]["content"]


def summarize_overall():
    """
    Generate overall summarizations from topic-wise summaries.
    """
    if os.path.exists(config.OVERALL_SUMMARIZATION_OUTPUT_PATH):
        print(f"Overall summarization output path {config.OVERALL_SUMMARIZATION_OUTPUT_PATH} already exists. Skipping overall summarization.")
        return

    os.makedirs(config.OVERALL_SUMMARIZATION_OUTPUT_PATH, exist_ok=True)

    for file in os.listdir(config.SUMMARIZATION_OUTPUT_PATH):
        if not file.endswith(".json"):
            continue

        print(f"Processing file: {file}")
        with open(os.path.join(config.SUMMARIZATION_OUTPUT_PATH, file), "r") as f:
            topic_transcript_summary = json.load(f)

        # collect only summaries per topic
        topic_summaries = {topic: content["summary"] for topic, content in topic_transcript_summary.items()}

        # call LLM
        response = llama3_overall(topic_summaries)

        # save
        output_file = os.path.join(
            config.OVERALL_SUMMARIZATION_OUTPUT_PATH,
            file.replace(".json", ".txt")
        )
        with open(output_file, "w") as f:
            f.write(response)
        print(f"Overall summary saved to {output_file}")


def main():
    summarize_overall()


if __name__ == "__main__":
    main()