from os.path import join
import os
from typing import List

import datasets
from datasets import Value, Sequence

from turntaking.dataload.dataset.scope.utils import (
    extract_dialog,
    extract_vad_list,
)
from turntaking.dataload.utils import (
    read_txt,
    read_json,
    repo_root,
)

logger = datasets.logging.get_logger(__name__)

EXTRACTED_PATH = "/ahc/work2/kazuyo-oni/projects/data/scope"

REL_AUDIO_PATH = join(
    repo_root(), "dataload/dataset/scope/files/relative_audio_path.json"
)
SPLIT_PATH = os.path.join(repo_root(), "dataload/dataset/scope/files")

_HOMEPAGE = "https://aclanthology.org/L18-1462/"
_DESCRIPTION = """
"""
_CITATION = """
    @inproceedings{yoshino-etal-2018-japanese,
        title = "{J}apanese Dialogue Corpus of Information Navigation and Attentive Listening Annotated with Extended {ISO}-24617-2 Dialogue Act Tags",
        author = "Yoshino, Koichiro  and
        Tanaka, Hiroki  and
        Sugiyama, Kyoshiro  and
        Kondo, Makoto  and
        Nakamura, Satoshi",
        booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
        month = may,
        year = "2018",
        address = "Miyazaki, Japan",
        publisher = "European Language Resources Association (ELRA)",
        url = "https://aclanthology.org/L18-1462",
    }
"""


FEATURES = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "user1_audio_path": Value("string"),
    "user2_audio_path": Value("string"),
    "vad": [
        [Sequence(Value("float"))],
    ],
}


class ScopeConfig(datasets.BuilderConfig):
    def __init__(
        self,
        train_sessions=None,
        val_sessions=None,
        test_sessions=None,
        min_word_vad_diff=0.05,
        ext=".wav",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ext = ext
        self.min_word_vad_diff = min_word_vad_diff
        self.train_sessions = (
            read_txt(os.path.join(SPLIT_PATH, "train.txt"))
            if train_sessions is None
            else train_sessions
        )
        self.val_sessions = (
            read_txt(os.path.join(SPLIT_PATH, "val.txt"))
            if val_sessions is None
            else val_sessions
        )
        self.test_sessions = (
            read_txt(os.path.join(SPLIT_PATH, "test.txt"))
            if test_sessions is None
            else test_sessions
        )


class Scope(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        ScopeConfig(
            name="default",
            description="Scope",
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            features=datasets.Features(FEATURES),
            supervised_keys=None,
        )

    def _split_generators(self, *args, **kwargs) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"sessions": self.config.train_sessions},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"sessions": self.config.val_sessions},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"sessions": self.config.test_sessions},
            ),
        ]

    def generate(self, sessions):
        sess_2_rel_audio_path = read_json(REL_AUDIO_PATH)
        for session in sessions:
            session = str(session)
            session_dir = join(EXTRACTED_PATH, session)
            dialog = extract_dialog(session, session_dir)
            vad = extract_vad_list(dialog)
            audio_path = join(session, sess_2_rel_audio_path[session])
            user1_audio_path = audio_path.replace("audio_mix", "audio_user1")
            user2_audio_path = audio_path.replace("audio_mix", "audio_user2")
            # omit words
            # dialog = remove_words_from_dialog(dialog)
            yield f"{session}", {
                "session": session,
                "audio_path": audio_path,
                "user1_audio_path": user1_audio_path,
                "user2_audio_path": user2_audio_path,
                "vad": vad,
            }

    def _generate_examples(self, sessions):
        # logger.info("generating examples from = %s", sessions)
        return self.generate(sessions)
