from os.path import join
import re
from pprint import pprint

from turntaking.dataload.utils import read_txt

OmitText = [
    "[silence]",
    "[noise]",
    "[vocalized-noise]",
]


# Only used once, json file is included in repo
def extract_audio_mapping(audio_root):
    map = {}
    for root, _, files in walk(audio_root):
        if len(files) > 0:
            rel_path = root.replace(audio_root, "")
            for f in files:
                if f.endswith(".wav") or f.endswith(".sph"):
                    # sw03815.{wav,sph} -> 3815
                    session = basename(f)
                    session = re.sub(r"^sw0(\w*).*", r"\1", session)
                    # sw03815.{wav,sph} ->sw03815
                    f_no_ext = re.sub(r"^(.*)\.\w*", r"\1", f)
                    map[session] = join(rel_path, f_no_ext)
    return map


def noxi_regexp(s):
    """
    Noxi annotation specific regexp.

    """
    # Noise
    s = re.sub(r"\[noise\]", "", s)
    s = re.sub(r"\[vocalized-noise\]", "", s)

    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # laughing and speech e.g. [laughter-yeah] -> yeah
    s = re.sub(r"\[laughter-(\w*)\]", r"\1", s)
    s = re.sub(r"\[laughter-(\w*\'*\w*)\]", r"\1", s)

    # Partial words: w[ent] -> went
    s = re.sub(r"(\w+)\[(\w*\'*\w*)\]", r"\1\2", s)
    # Partial words: -[th]at -> that
    s = re.sub(r"-\[(\w*\'*\w*)\](\w+)", r"\1\2", s)

    # restarts
    s = re.sub(r"(\w+)-\s", r"\1 ", s)
    s = re.sub(r"(\w+)-$", r"\1", s)

    # Pronounciation variants
    s = re.sub(r"(\w+)\_\d", r"\1", s)

    # Mispronounciation [splace/space] -> space
    s = re.sub(r"\[\w+\/(\w+)\]", r"\1", s)

    # Coinage. remove curly brackets... keep word
    s = re.sub(r"\{(\w*)\}", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end


def extract_vad_list(anno):
    vad = [[], []]
    for channel in [0, 1]:
        for utt in anno[channel]:
            s, e = utt["start"], utt["end"]
            vad[channel].append((s, e))
    return vad


# def extract_word_level_annotations(session, speaker, session_dir, apply_regexp=True):
#     def remove_multiple_whitespace(s):
#         s = re.sub(r"\t", " ", s)
#         return re.sub(r"\s\s+", " ", s)

#     # Load word-level annotations
#     words_filename = "sw" + session + speaker + "-ms98-a-word.text"
#     words_list = read_txt(join(session_dir, words_filename))

#     # process word-level annotation
#     word_dict = {}
#     for word_row in words_list:
#         word_row = remove_multiple_whitespace(word_row).strip()
#         try:
#             idx, wstart, wend, word = word_row.split(" ")
#         except Exception as e:
#             print("word_row: ", word_row)
#             print("word_split: ", word_row.split(" "))
#             print(e)
#             input()
#             assert False

#         if apply_regexp:
#             word = noxi_regexp(word)

#         if not (word in OmitText or word == ""):
#             if idx in word_dict:
#                 word_dict[idx].append(
#                     {"text": word, "start": float(wstart), "end": float(wend)}
#                 )
#             else:
#                 word_dict[idx] = [
#                     {"text": word, "start": float(wstart), "end": float(wend)}
#                 ]
#     return word_dict


def combine_speaker_utterance_and_words(
    session, speaker, session_dir, apply_regexp=True
):
    """Combines word- and utterance-level annotations"""
    # Read word-level annotation and format appropriately
    # word_dict = extract_word_level_annotations(
    #     session, speaker, session_dir, apply_regexp=apply_regexp
    # )

    # Read utterance-level annotation
    if speaker == "user1":
        trans_filename = "vad_user1.txt"
    else:
        trans_filename = "vad_user2.txt"
    trans_list = read_txt(join(session_dir, trans_filename))

    # correct channels for wavefiles
    speaker = 0 if speaker == "user1" else 1

    # Combine word-/utterance- level annotations
    utterances = []
    for row in trans_list:
        # utt_start/end are padded so we use exact word timings
        utt_idx, utt_start, utt_end, *words = row.split(" ")

        if not (words[0] in OmitText and len(words) == 1):  # only noise/silence
            # wd = word_dict.get(utt_idx, None)
            # if wd is None:
            #     continue

            # words = " ".join(words)
            # if apply_regexp:
            #     words = noxi_regexp(words)

            utterances.append(
                {
                    # "text": words,
                    # "words": wd,
                    "start": utt_start,
                    "end": utt_end,
                }
            )
    return utterances


def extract_dialog(session, session_dir, apply_regexp=True):
    """Extract the annotated dialogs based on config `name`"""
    # Config settings

    # Speaker A: original name in annotations
    a_utterances = combine_speaker_utterance_and_words(
        session,
        speaker="user1",
        session_dir=session_dir,
        apply_regexp=apply_regexp,
    )

    # Speaker B: original name in annotations
    b_utterances = combine_speaker_utterance_and_words(
        session,
        speaker="user2",
        session_dir=session_dir,
        apply_regexp=apply_regexp,
    )
    return [a_utterances, b_utterances]


def remove_words_from_dialog(dialog):
    new_dialog = [[], []]
    for channel in [0, 1]:
        for utt in dialog[channel]:
            new_dialog[channel].append(
                {
                    "text": utt["text"],
                    "start": utt["start"],
                    "end": utt["end"],
                }
            )
    return new_dialog


if __name__ == "__main__":
    from os import listdir

    extracted_path = "/ahc/work2/kazuyo-oni/projects/data/eald"
    session = "student_01"

    session = str(session)
    session_dir = join(extracted_path, session)
    print(listdir(session_dir))
    dialog = extract_dialog(session, session_dir)
    pprint(dialog)
    vad = extract_vad_list(dialog)
    pprint(vad)
