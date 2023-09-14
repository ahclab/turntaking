from os.path import join, expanduser
from pprint import pprint
from datasets import load_dataset
from turntaking.dataload.utils import repo_root, read_txt
from datasets import Dataset, DatasetDict
import itertools

AUDIO_DIR = "/ahc/work2/kazuyo-oni/projects/data/switchboard/audio"

DATASET_SCRIPT = join(repo_root(), "dataload/dataset/switchboard/switchboard.py")
EXT = ".wav"

SPLIT_PATH = join(repo_root(), "dataload/dataset/switchboard/files")

def load_switchboard(
    split="train",
    audio_root=AUDIO_DIR,
    ext=EXT,
    train_files=None,
    val_files=None,
    test_files=None,
):

    def session_generator(
        dset,
        split,
        train_files=None,
        val_files=None,
        test_files=None,
    ):
        train_sessions = (
            read_txt(join(SPLIT_PATH, "train.txt"))
            if train_files is None
            else read_txt(train_files)
        )
        val_sessions = (
            read_txt(join(SPLIT_PATH, "val.txt"))
            if val_files is None
            else read_txt(val_files)
        )
        test_sessions = (
            read_txt(join(SPLIT_PATH, "test.txt"))
            if test_files is None
            else read_txt(test_files)
        )

        if split == "train":
            train_dset = dset.filter(lambda example: example["session"] in train_sessions)
            return train_dset
        elif split == "validation":
            val_dset = dset.filter(lambda example: example["session"] in val_sessions)
            return val_dset
        elif split == "test":
            test_dset = dset.filter(lambda example: example["session"] in test_sessions)
            return test_dset
        else:
            print("split Error")
            exit()

        #dataset = DatasetDict({
        #    "train":train_dset,
        #    "validation":val_dset,
        #    "test":test_dset,
        #    })

        #return dataset

    if split == "val":
        split = "validation"

    def process_and_add_name(examples):
        examples["dataset_name"] = "switchboard"
        if audio_root is not None:
            examples["audio_path"] = join(audio_root, examples["audio_path"] + ext)

        return examples

    dset = load_dataset(DATASET_SCRIPT,name="default",split="train+validation+test")
    dset = dset.map(process_and_add_name)

    dataset = session_generator(
        dset,
        split,
        train_files,
        val_files,
        test_files,
        )

    #print(dataset)
    return dataset
