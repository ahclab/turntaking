from os.path import join
from datasets import load_dataset
from turntaking.dataload.utils import repo_root, read_txt

AUDIO_DIR = "/ahc/work2/kazuyo-oni/projects/data/scope"

DATASET_SCRIPT = join(repo_root(), "dataload/dataset/scope/scope.py")
AUDIO_EXT = ".wav"

SPLIT_PATH = join(repo_root(), "dataload/dataset/scope/files")


def load_scope(
    split="train",
    audio_root=AUDIO_DIR,
    audio_ext=AUDIO_EXT,
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
            train_dset = dset.filter(
                lambda example: example["session"] in train_sessions
            )
            return train_dset
        elif split == "validation":
            val_dset = dset.filter(lambda example: example["session"] in val_sessions)
            return val_dset
        elif split == "test":
            test_dset = dset.filter(lambda example: example["session"] in test_sessions)
            return test_dset
        else:
            print("split Error")
            exit(1)

    if split == "val":
        split = "validation"

    def process_and_add_name(examples):
        examples["dataset_name"] = "scope"
        if audio_root is not None:
            examples["audio_path"] = join(
                audio_root, examples["audio_path"] + audio_ext
            )
            examples["user1_audio_path"] = join(
                audio_root, examples["user1_audio_path"] + audio_ext
            )
            examples["user2_audio_path"] = join(
                audio_root, examples["user2_audio_path"] + audio_ext
            )

        return examples

    dset = load_dataset(DATASET_SCRIPT, name="default", split="train+validation+test")
    dset = dset.map(process_and_add_name)

    dataset = session_generator(
        dset,
        split=split,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
    )

    return dataset
