import os
import json
import re
import numpy as np
from lxml import etree
from nltk.metrics import segmentation

try:
    from .. import config       # works when run from project_root
except ImportError:
    import config  



NS = {'nite': 'http://nite.sourceforge.net/'}


def find_the_best_match(transcription_list, start_sentence_list, ref=False):
    """
    Match each sentence fragment to a transcription sentence index.
    Uses simple word-overlap scoring.
    """
    index_list = []
    for value in start_sentence_list:
        start_sentence = re.split(
            r'''(?<!\w)'(?!\w)|[!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~]''',
            value,
        )
        start_sentence = [s.strip() for s in start_sentence if s.strip()]
        if not start_sentence:
            continue

        longest = max(start_sentence, key=len)
        if len(longest.split()) < 3:
            continue

        best_idx, best_score = -1, -1
        longest_words = set(longest.lower().split())
        for idx, t in enumerate(transcription_list):
            t_words = set(t.lower().split())
            score = len(longest_words & t_words)
            if score > best_score:
                best_idx, best_score = idx, score

        if ref:
            index_list.append(best_idx)
        else:
            if best_score < len(longest_words) * 0.5:
                print(
                    f"Skipping '{longest}' (low match with '{transcription_list[best_idx]}')."
                )
                continue
            index_list.append(best_idx)
    return index_list


def load_ref_pred_json():
    """
    Build binary reference and prediction boundary arrays from
    AMI XML reference and model prediction JSONs.
    """
    compare_dict = {}
    for file in os.listdir(config.PREDICTION_FILE_PATH):
        if not file.endswith('.txt'):
            continue

        file_id = file.split('.')[0]
        prediction_file = os.path.join(config.SEGMENTATION_OUTPUT_PATH, file.replace('.txt', '.json'))
        print(f'Processing prediction file: {prediction_file}')
        reference_file = os.path.join(config.REFERENCE_FILE_PATH, file_id + '.topic.xml')
        transcript_file = os.path.join(config.TRANSCRIPTION_FILE_PATH, file_id + '.json')

        print(f'--- Processing {file_id} ---')

        with open(transcript_file, 'r') as f:
            transcript = json.load(f)

        pred_list = np.zeros((len(transcript.keys()),), dtype=int)
        ref_list = np.zeros((len(transcript.keys()),), dtype=int)

        # Parse reference XML
        tree = etree.parse(reference_file)
        root = tree.getroot()
        topics_start_word_id = list(transcript.values())

        for topic in root.findall('.//topic'):
            for child in topic.findall('.//nite:child', namespaces=NS):
                href = child.get('href')
                _, fragment = href.split('#')
                if '..' in fragment:
                    start_id, _ = fragment.replace('id(', '').replace(')', '').split('..')
                    if start_id in topics_start_word_id:
                        idx = topics_start_word_id.index(start_id)
                        ref_list[idx] = 1
                        break

        # Parse model predictions
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        for value in predictions:
            pred_list[value['start_sentence']] = 1

        compare_dict[file_id] = {
            'ref': ref_list.tolist(),
            'pred': pred_list.tolist(),
        }

    # Save results
    os.makedirs(os.path.dirname(config.OUTPUT_COMPARE_JSON), exist_ok=True)
    with open(config.OUTPUT_COMPARE_JSON, 'w') as f:
        json.dump(compare_dict, f, indent=2)

    return compare_dict


def metrics_pk(compare_dict):
    """
    Compute segmentation Pk and WindowDiff errors.
    """
    pk_error, window_error = [], []
    for file_id, values in compare_dict.items():
        print(f"Processing file ID: {file_id}")
        ref = [str(x) for x in values['ref']]
        pred = [str(x) for x in values['pred']]

        n_boundaries = ref.count("1")
        if n_boundaries == 0:
            continue
        k = int(round(len(ref) / (n_boundaries * 2.0)))

        pk_err = segmentation.pk(ref, pred, k=None, boundary='1')
        wd_err = segmentation.windowdiff(ref, pred, k, boundary='1')
        pk_error.append(pk_err)
        window_error.append(wd_err)

    overall_pk_err = np.mean(pk_error)
    overall_window_err = np.mean(window_error)
    print(f"\nOverall PK Error: {overall_pk_err:.4f}, Overall Window Error: {overall_window_err:.4f}")
    return overall_pk_err, overall_window_err


def main():
    compare_dict = load_ref_pred_json()
    metrics_pk(compare_dict)


if __name__ == '__main__':
    main()