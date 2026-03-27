"""
Generate utterance-level transcripts for the AMI corpus using manual segments.

For each meeting, this script:
- Reads manual words files:   dataset/ami_public_manual_1.6.2/words/{meeting_id}.*.words.xml
- Reads manual segments files: dataset/ami_public_manual_1.6.2/segments/{meeting_id}.*.segments.xml
- Produces a JSON file per meeting at dataset/utterance_json/{meeting_id}.utterances.json

JSON format:
{
    "0": {
        "text": "first utterance text ...",
        "start_time": 12.34,
        "end_time": 15.67,
        "speakers": ["A"]
    },
    "1": {
        "text": "second utterance text ...",
        "start_time": ...,
        "end_time": ...,
        "speakers": ["B"]
    },
    ...
}
"""

from lxml import etree
from operator import itemgetter
import glob
import os
import json
import subprocess
from pathlib import Path
import json


# NITE namespace
NS = {'nite': 'http://nite.sourceforge.net/'}

def run_ffmpeg(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def make_personal_av(meeting_id, camera, video_file, audio_file, out_dir):
    """
    Create a 'personal' AV file combining closeup video + mixed lapel audio.
    Returns path to the combined file.
    """
  
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # if not video_file.exists():
    #     raise FileNotFoundError(f"Video not found: {video_file}")
    # if not audio_file.exists():
    #     raise FileNotFoundError(f"Audio not found: {audio_file}")

    # Output: ES2006b.Closeup4.personal.mp4
    out_file = out_dir / f"{meeting_id}.{camera}.mp4"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_file),   # video source
        "-i", str(audio_file),   # audio source
        "-map", "0:v:0",         # take video from first input
        "-map", "1:a:0",         # take audio from second input
        "-c:v", "copy",          # copy video stream (no re-encode)
        "-c:a", "aac",           # encode audio to AAC for mp4
        "-shortest",             # stop at shorter of audio/video
        str(out_file),
    ]
    run_ffmpeg(cmd)
    return out_file

def cut_segment(av_file, start_time, end_time, out_path):
    """
    Cut a [start_time, end_time] segment from av_file into out_path.
    """
    duration = end_time - start_time
    if duration <= 0:
        return

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_time:.3f}",
        "-i", str(av_file),
        "-t", f"{duration:.3f}",
        "-c", "copy",           # copy both audio + video
        str(out_path),
    ]
    run_ffmpeg(cmd)

def cut_personal_video_from_json(meeting_id, camera, json_path, personal_av_path, out_dir):
    """
    For a given meeting + camera (e.g. Closeup4), read the utterance JSON and
    cut the personal AV into utterance-wise clips where speakers contain that camera.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        utterances = json.load(f)

    for utt_id, utt in utterances.items():
        speakers = utt.get("speakers", [])
        if camera not in speakers:
            continue  # not this camera

        start = float(utt["start_time"])
        end = float(utt["end_time"])
        text = utt["text"]

        # e.g. ES2006b.Closeup4_utt00078.mp4
        out_name = f"{meeting_id}.{camera}_utt{int(utt_id)}.mp4"
        out_path = out_dir / out_name

        cut_segment(personal_av_path, start, end, out_path)

        print(f"Cut {out_name}: {start:.3f}–{end:.3f}  \"{text[:60]}...\"")

def load_participant_camera_map(meetings_xml_path, meeting_code):
    """
    Given the path to corpusResources/meetings.xml and a meeting code,
    return a dict mapping AMI participant letters (A/B/C/D) to closeup cameras.

    `meeting_code` can be either:
      - the AMI observation id, e.g. "ES2006b"
      - the internal nite:id, e.g. "meet_57"
    """
    if not os.path.exists(meetings_xml_path):
        raise FileNotFoundError(f"meetings.xml not found at: {meetings_xml_path}")

    tree = etree.parse(meetings_xml_path)
    root = tree.getroot()

    target_meeting = None

    for m in root.findall(".//meeting"):
        obs_id = m.get("observation")                 # e.g. "ES2006b"
        nite_id = m.get("{http://nite.sourceforge.net/}id")  # e.g. "meet_57"

        if meeting_code == obs_id or meeting_code == nite_id:
            target_meeting = m
            break

    if target_meeting is None:
        raise ValueError(f"Meeting '{meeting_code}' not found in meetings.xml")

    mapping = {}

    # Your snippet structure:
    # <speaker ... nxt_agent="A" camera="Closeup2" ... />
    for spk in target_meeting.findall("speaker"):
        agent = spk.get("nxt_agent")    # A / B / C / D
        camera = spk.get("camera")      # Closeup1 / Closeup2 / ...
        if agent and camera:
            mapping[agent] = camera

    return mapping

def manual_parse_words_and_silences(words_path):
    """
    Parse a manual AMI words.xml file into:
    - list of word dicts (in file order)
    - dict mapping word_id -> word_info
    """
    tree = etree.parse(words_path)
    root = tree.getroot()

    words = []
    words_dict = {}

    for word_elem in root:
        word_id = word_elem.get('{http://nite.sourceforge.net/}id')

        # If the word element does not have a start time, use the end time of the last word
        if word_elem.get('starttime') is None:
            start = words[-1]['end'] if words else 0.0
        else:
            start = float(word_elem.get('starttime'))

        if word_elem.get('endtime') is None:
            end = start
        else:
            end = float(word_elem.get('endtime'))

        speaker = word_elem.get('speaker')
        text = word_elem.text.strip() if word_elem.text else ''

        word_info = {
            'id': word_id,
            'start': start,
            'end': end,
            'speaker': speaker,
            'text': text,
            'punctuation': word_elem.get('punc', ''),  # Optional punctuation attribute
            'word_type': word_elem.tag                # Optional type attribute
        }

        words.append(word_info)
        words_dict[word_id] = word_info

    return words, words_dict


def combine_words_from_multiple_files(file_paths):
    """
    Combines word entries from multiple words.xml files.
    Returns:
        - a list of all words sorted by start time
        - a dict mapping word_id -> word_info
    """
    all_words = []
    all_words_dict = {}

    for file_path in file_paths:
        words, words_dict = manual_parse_words_and_silences(file_path)
        all_words.extend(words)

        for word_id, word in words_dict.items():
            if word_id not in all_words_dict:
                all_words_dict[word_id] = word
            else:
                # If the word already exists, merge the text and update the end time
                existing_word = all_words_dict[word_id]
                existing_word['text'] += ' ' + word['text']
                existing_word['end'] = max(existing_word['end'], word['end'])
                existing_word['punctuation'] = existing_word.get('punctuation', '') or word.get('punctuation', '')
                existing_word['word_type'] = existing_word.get('word_type', '') or word.get('word_type', '')

    all_words_sorted = sorted(all_words, key=itemgetter('start'))

    return all_words_sorted, all_words_dict


def parse_manual_segments(segments_path):
    """
    Parse a manual AMI segments.xml file into a list of segments with:
        segment_id, start_id, end_id, start_time, end_time
    """
    tree = etree.parse(segments_path)
    root = tree.getroot()

    segments = []
    for seg in root.findall('segment'):
        segment_id = seg.get('nite:id')
        start_time = float(seg.get('transcriber_start'))
        end_time = float(seg.get('transcriber_end'))

        child = seg.find('nite:child', namespaces=NS)
        if child is None:
            continue

        # href format: file#id(start)..id(end) or file#id(x)
        href = child.get('href')
        _, fragment = href.split('#')

        if '..' in fragment:
            start_id, end_id = fragment.replace('id(', '').replace(')', '').split('..')
        else:
            start_id = fragment.replace('id(', '').replace(')', '')
            end_id = start_id

        segments.append({
            'segment_id': segment_id,
            'start_id': start_id,
            'end_id': end_id,
            'start_time': start_time,
            'end_time': end_time
        })

    print(f"Parsed {len(segments)} segments from {os.path.basename(segments_path)}")
    return segments


def combine_segs_from_multiple_files(file_paths, words_dict, speaker):
    """
    Combines segment entries from multiple segments.xml files and builds utterance-level text.

    Returns:
        - list of segment dicts (with added 'text' and 'speakers' fields)
        - transcript_json: {utterance_index: {
                                "text": str,
                                "start_time": float,
                                "end_time": float,
                                "speakers": [str, ...]
                            }}
    """
    all_seg = []
    transcript_json = {}

    # Collect all segments from all files
    for file_path in file_paths:
        segs = parse_manual_segments(file_path)
        all_seg.extend(segs)

    print(f"Total segments combined: {len(all_seg)}")

    # Sort segments by start time
    all_seg_sorted = sorted(all_seg, key=itemgetter('start_time'))

    json_key = 0
    for i, seg in enumerate(all_seg_sorted):
        text = ''
        speakers = set()

        # Extract numeric suffix from word ids like 'EN2001b.A.words0'
        ind_start = seg['start_id'].split('.words')[1]
        ind_end = seg['end_id'].split('.words')[1]
        base_id = seg['start_id'].split('.words')[0]

        for ind in range(int(ind_start), int(ind_end) + 1):
            words_dict_key = base_id + '.words' + str(ind)

            if words_dict_key not in words_dict:
                continue

            word = words_dict[words_dict_key]

            # Track speakers present in this utterance
            if word.get('speaker'):
                speakers.add(word['speaker'])

            # Build text with punctuation merged to previous token
            if word.get('punctuation'):
                # Attach punctuation to the previous token (remove trailing space)
                text = text[:-1] + word['text'] + ' '
            else:
                text += word['text'] + ' '

        utt_text = text.strip()
        utt_speakers = sorted(list(speakers))

        all_seg_sorted[i]['text'] = utt_text
        all_seg_sorted[i]['speakers'] = utt_speakers

        if utt_text:
            transcript_json[str(json_key)] = {
                "text": utt_text,
                "start_time": seg['start_time'],
                "end_time": seg['end_time'],
                "speakers": speaker
            }
            json_key += 1

    return all_seg_sorted, transcript_json


if __name__ == '__main__':
    qulified_dir = "./ami/amicorpus"
    dirs = os.listdir(qulified_dir)
    dirs.remove('.DS_Store')
    print(dirs)
    base_dir = "./ami/ami_public_manual_1.6.2"
    meeting_info_path = os.path.join(base_dir, "corpusResources/meetings.xml")
    words_dir = os.path.join(base_dir, "words")
    segments_dir = os.path.join(base_dir, "segments")

    output_dir = "./ami/utterance_json"
    os.makedirs(output_dir, exist_ok=True)
    for meeting_id in dirs:
        partipants_id = load_participant_camera_map(meeting_info_path, meeting_id)
        print(f"Processing meeting: {partipants_id}")
        for key, value in partipants_id.items():
            print(f"  Participant {key} -> Camera {partipants_id[key]}")
        

        # All word and segment files for this meeting
            word_file_paths = glob.glob(os.path.join(words_dir, f"{meeting_id}.{key}.words.xml"))
            utterance_file_paths = glob.glob(os.path.join(segments_dir, f"{meeting_id}.{key}.segments.xml"))

            if not word_file_paths:
                print("  No .words.xml files found. Skipping.")
                continue

            if not utterance_file_paths:
                print("  No .segments.xml files found. Skipping.")
                continue

            # Combine words and segments → utterance-level transcripts
            _, combined_words_dict = combine_words_from_multiple_files(word_file_paths)
            _, transcript_json = combine_segs_from_multiple_files(utterance_file_paths, combined_words_dict, value)

        # Save utterance-level transcript JSON
            out_path = os.path.join(output_dir, f"{meeting_id}.utterances.json")
            with open(out_path, 'w') as f:
                json.dump(transcript_json, f, indent=4, ensure_ascii=False)

            print(f"  Saved {len(transcript_json)} utterances to {out_path}")


            video_dir = f"./ami/amicorpus/{meeting_id}/video/{meeting_id}.{value}.avi"
            if not os.path.exists(video_dir):

                audio_dir_headset = f"./ami/amicorpus/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"
                audio_dir_lapel = f"./ami/amicorpus/{meeting_id}/audio/{meeting_id}.Mix-Lapel.wav"
                if os.path.exists(audio_dir_headset):
                    audio_dir = audio_dir_headset
                else:
                    audio_dir = audio_dir_lapel
                # print("--------------",audio_dir)
                personal_out_dir = "./ami/video_with_audio"
                clips_out_dir = "./ami/personal_clips"

                # 1) make personal AV
    
                personal_av = make_personal_av(
                    meeting_id=meeting_id,
                    camera=value,
                    video_file=video_dir,
                    audio_file=audio_dir,
                    out_dir=personal_out_dir
                )



                cut_personal_video_from_json(
                    meeting_id=meeting_id,
                    camera=value,
                    json_path=out_path,
                    personal_av_path=personal_av,
                    out_dir=os.path.join(clips_out_dir, f"{meeting_id}_{value}")
                )
        