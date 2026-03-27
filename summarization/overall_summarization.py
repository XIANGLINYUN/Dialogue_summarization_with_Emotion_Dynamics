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
        "You are an expert in summarization, and you know how to generate informative summaries with emotion information. "
        "You only know the following emotions: anger, happiness, excited, sadness, frustration, fear, surprise, and neutral."
    )


    user_message_1shot = '''
I have a JSON file containing several topics and corresponding summaries with emotion information, which are segmented from an entire conversation.
Please perform summarization on the following sub topics:

{
  "Wrong Line": {
    "summary": "Speaker 1 is frustrated as they're stuck in the wrong line for an ID. Speaker 2 tries to help, but also shows frustration at being unable to assist, while remaining neutral about the situation."
  },
  "Explanation and Direction": {
    "summary": "Speaker 2 is frustrated while trying to help someone with an issue, speaker 1 also shows frustration when asking for the right information."
  },
  "Request for Accommodation": {
    "summary": "Speaker 1 is frustrated while trying to navigate the situation at the front of the line, asking Speaker 2 to put them there with a mix of frustration and neutrality. Meanwhile, Speaker 2 is also frustrated but tries to be helpful, ultimately directing Speaker 1 to the correct line with a sense of other emotions."
  },
  "Frustration and Impatience": {
    "summary": "Speaker 1 is frustrated while trying to get help from Speaker 2, who seems unable or unwilling to assist."
  },
  "Accusation of Lack of Compassion": {
    "summary": "Speaker 2, frustrated and angry, tries to manage a situation at the D.M.V. where Speaker 1 has been waiting in line for an hour. Speaker 1 expresses frustration as well, feeling misinformed by staff about which line to stand in. The conversation becomes increasingly heated and tense."
  },
  "Request for Assistance": {
    "summary": "Speaker 1 is frustrated and angry while trying to navigate a confusing system, with Speaker 2 responding with equal frustration and anger. Despite the tension, neither party finds resolution until Speaker 2 finally explains the strict system in place, leaving Speaker 1 with a sense of resignation."
  },
  "Confrontation and Concessions": {
    "summary": "Speaker 1 is frustrated and angry while trying to resolve an issue, asking if the manager can be spoken with and being straightforward about their concerns. The manager, Speaker 2, also shows frustration and anger, but attempts to remain neutral towards the end, stating 'There is nothing I can do for you' and directing Speaker 1 to stand in line two A."
  }
}

Write an overall summary capturing the trajectory of the conversation. 
Summarize the context-level information, also integrate the speakers' emotions trajectory, keep the speaker information. Make the summary informative and concise.

Restrict all the emotion-related expressions in the summary within the following ones: 
anger, happiness, excited, sadness, frustration, fear, surprise, and neutral.

Return only the result with plain text. Final output format: 
Summary: #summary for the entire conversation#

Please make sure that you make only one summarization for the given sub topic summaries.  
Do not include any extra text or explanation—only return plain text.
'''
    assistant_message = "Speaker1 begins neutral, seeking help but grows frustrated and angry when faced with rigid procedures. Speaker2, initially neutral, becomes increasingly frustrated and angry while enforcing rules. Their frustration builds into open conflict, with neither finding resolution. The exchange ends in sadness and lingering frustration, leaving both emotionally exhausted."

    user_message = f'''
I have a JSON file containing several topics and corresponding summaries with emotion information, which are segmented from an entire conversation.
Please perform summarization on the following sub topics:

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
                "content":user_message_1shot 
            },
        {
                "role": "assistant",
                "content": assistant_message
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
            + user_message_1shot.strip()
            + "\n\n=== ASSISTANT MESSAGE ===\n"
            + assistant_message.strip()
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