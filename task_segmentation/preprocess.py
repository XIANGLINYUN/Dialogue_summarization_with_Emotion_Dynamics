import os
import json
from collections import Counter
try:
    from .. import config       # works when run from project_root
except ImportError:
    import config  


def ensure_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def generate_json_with_emotion():
    """
    Generate JSON files with transcript and emotion data.
    """
    script_dict = {}
    script_emotion_dict = {}
    script_json_emotion_filtered = {}

    for file in os.listdir(config.TRANSCRIPT_FILE_PATH):
        if not file.endswith('.txt'):
            continue

        script_id = (file.split('_')[0] + '_' + file.split('_')[1]).replace('.txt', '')
        script_dict.setdefault(script_id, {})
        script_emotion_dict.setdefault(script_id, {})

        transcript_file = os.path.join(config.TRANSCRIPT_FILE_PATH, file)
        emotion_file = os.path.join(config.EMOTION_FILE_PATH, file)

        with open(transcript_file, 'r') as f:
            transcript = f.read()
        with open(emotion_file, 'r') as f:
            emotion = f.read()

            emotion_json = {}
            for line in emotion.split('\n\n')[1:]:
                if 'Ses' not in line:
                    break
                info = line.split('\n')[0].split('\t')
                emo1 = line.split('\n')[1].split(':')[1].strip().split(';')[0] if '()' in line else line.split('(')[1].split(')')[0].strip()
                emo2 = line.split('\n')[2].split(':')[1].strip().split(';')[0] if '()' in line else line.split('(')[1].split(')')[0].strip()
                emo3 = line.split('\n')[3].split(':')[1].strip().split(';')[0] if '()' in line else line.split('(')[1].split(')')[0].strip()
                emotion_json[info[1]] = {
                    'time': info[0],
                    'avg_emotion': info[2],
                    'VAD': info[3],
                    'emotions': [emo1, emo2, emo3],
                }

        json_index = 0
        for line in transcript.split('\n'):
            if not line.strip():
                continue
            utterance = line.split(':')[1]
            speaker = line.split(' ')[0]

            script_dict[script_id][json_index] = utterance.strip()
            if speaker in emotion_json:
                script_emotion_dict[script_id][json_index] = {
                    'utterance': utterance.strip(),
                    'emotion': emotion_json[speaker]['emotions'],
                    'speaker': 'Speaker1' if 'F' in speaker.split('_')[-1] else 'Speaker2'
                }
            else:
                script_emotion_dict[script_id][json_index] = {
                    'utterance': utterance.strip(),
                    'emotion': None,
                    'speaker': 'Speaker1' if 'F' in speaker.split('_')[-1] else 'Speaker2'
                }
            json_index += 1

    # Save transcripts
    for script_id, sentences in script_dict.items():
        if sentences:
            with open(os.path.join(config.PROCESSED_TRANSCRIPT_PATH, f"{script_id}.json"), 'w') as f:
                json.dump(sentences, f, indent=4)

    # Save transcripts with emotions and filtered versions
    for script_id, sentences in script_emotion_dict.items():
        script_json_emotion_filtered[script_id] = {}
        if sentences:
            with open(os.path.join(config.PROCESSED_TRANSCRIPT_EMOTION_PATH, f"{script_id}.json"), 'w') as f:
                json.dump(sentences, f, indent=4)

            for index, sentence in sentences.items():
                filtered = {'utterance': sentence['utterance'], 'speaker': sentence['speaker']}
                if sentence['emotion']:
                    counts = Counter(sentence['emotion'])
                    most_common = counts.most_common(1)
                    if most_common and most_common[0][1] > 1:
                        filtered['emotion'] = [most_common[0][0]]
                    else:
                        filtered['emotion'] = sentence['emotion']
                else:
                    filtered['emotion'] = None
                script_json_emotion_filtered[script_id][index] = filtered



            with open(os.path.join(config.PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH, f"{script_id}.json"), 'w') as f:
                json.dump(script_json_emotion_filtered[script_id], f, indent=4)

    return script_dict, script_emotion_dict


def main():
    if os.path.exists(config.PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH):
        print(f"Processed output path {config.PROCESSED_OUTPUT_PATH} already exists. Skipping processing.")
        return

    ensure_directories([
        config.PROCESSED_OUTPUT_PATH,
        config.PROCESSED_TRANSCRIPT_PATH,
        config.PROCESSED_TRANSCRIPT_EMOTION_PATH,
        config.PROCESSED_TRANSCRIPT_FILTERED_EMOTION_PATH
    ])

    generate_json_with_emotion()


if __name__ == "__main__":
    main()
