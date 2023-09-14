from os.path import join
import os
from typing import List

import datasets
from datasets import Value, Sequence

from turntaking.dataload.dataset.noxi.utils import (
    extract_dialog,
    extract_vad_list,
)
from turntaking.dataload.utils import (
    read_txt,
    read_json,
    repo_root,
)

logger = datasets.logging.get_logger(__name__)

EXTRACTED_PATH = "/ahc/work2/kazuyo-oni/projects/data/noxi"

REL_AUDIO_PATH = join(
    repo_root(), "dataload/dataset/noxi/files/relative_audio_path.json"
)
SPLIT_PATH = os.path.join(repo_root(), "dataload/dataset/noxi/files")

_HOMEPAGE = "https://noxi.aria-agent.eu"
_DESCRIPTION = """
NoXi annotations in a convenient format
TODO
"""
_CITATION = """
@inproceedings{Godfrey92,
    author = {A. Cafaro, J. Wagner, T. Baur, S. Dermouche, M. Torres Torres, C. Pelachaud, E. André, and M. Valstar,},
    title = {The NoXi database: multimodal recordings of mediated novice-expert interactions},
    year = {2017},
    isbn = {978-1-4503-5543-8},
    publisher = {Association for Computing Machinery},
    address = {USA},
    booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction},
    pages = {350–359},
    numpages = {9},
    location = {New York, NY},
    series = {ICMI '17}
}
"""


FEATURES = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "expert_audio_path": Value("string"),
    "novice_audio_path": Value("string"),
    "multimodal_expert_path": Value("string"),
    "multimodal_novice_path": Value("string"),
    "vad": [
        [Sequence(Value("float"))],
    ],
}


class NoxiConfig(datasets.BuilderConfig):
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


class Noxi(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        NoxiConfig(
            name="default",
            description="NoXi dataset",
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
            expert_audio_path = audio_path.replace("audio_mix", "audio_expert")
            novice_audio_path = audio_path.replace("audio_mix", "audio_novice")
            mutimodal_expert_path = audio_path.replace("audio_mix", "non_varbal_expert")
            mutimodal_novice_path = audio_path.replace("audio_mix", "non_varbal_novice")
            # omit words
            # dialog = remove_words_from_dialog(dialog)
            yield f"{session}", {
                "session": session,
                "audio_path": audio_path,
                "expert_audio_path": expert_audio_path,
                "novice_audio_path": novice_audio_path,
                "multimodal_expert_path": mutimodal_expert_path,
                "multimodal_novice_path": mutimodal_novice_path,
                "vad": vad,
            }

    def _generate_examples(self, sessions):
        # logger.info("generating examples from = %s", sessions)
        return self.generate(sessions)
