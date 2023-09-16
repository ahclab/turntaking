from os.path import join
from datasets import load_dataset
from turntaking.dataload.utils import repo_root, read_txt

AUDIO_DIR = "/ahc/work2/kazuyo-oni/projects/data/noxi"
MULTIMODAL_DIR = "/ahc/work2/kazuyo-oni/projects/data/noxi"

DATASET_SCRIPT = join(repo_root(), "dataload/dataset/noxi/noxi.py")
NORM_JSON = join(repo_root(), "dataload/dataset/noxi/files/normalize.json")
AUDIO_EXT = ".wav"
MULTIMODAL_EXT = ".csv"

SPLIT_PATH = join(repo_root(), "dataload/dataset/noxi/files")


def load_noxi(
    split="train",
    audio_root=AUDIO_DIR,
    multimodal_root=MULTIMODAL_DIR,
    audio_ext=AUDIO_EXT,
    multimodal_ext=MULTIMODAL_EXT,
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
        examples["dataset_name"] = "noxi"
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
        if multimodal_root is not None:
            examples["multimodal_user1_path"] = join(
                multimodal_root, examples["multimodal_user1_path"] + multimodal_ext
            )
            examples["multimodal_user2_path"] = join(
                multimodal_root, examples["multimodal_user2_path"] + multimodal_ext
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

    # if split == "train":
    #     gaze_x = []
    #     gaze_y = []
    #     head = []
    #     pose = []
    #     normalization = {}
    #     for path in dataset["multimodal_path"]:
    #         df = pd.read_csv(path)
    #         gaze_x += df["gaze_x"].values.tolist()
    #         gaze_y += df["gaze_y"].values.tolist()
    #         head += df["head"].values.tolist()
    #         pose += df["pose"].values.tolist()
    #     # print(max(head))
    #     # print(statistics.mean(head))
    #     # print(statistics.mean(head))
    #     # print(statistics.median(head))
    #     # print(max(pose))
    #     # normalization["gaze_x_mean"] = torch.mean(torch.tensor(gaze_x), 0)
    #     # ormalization["gaze_y_mean"] = torch.mean(torch.tensor(gaze_y), 0)
    #     # normalization["gaze_x_var"] = torch.var(torch.tensor(gaze_x), 0)
    #     # ormalization["gaze_y_var"] = torch.var(torch.tensor(gaze_y), 0)
    #     normalization["pose"] = max(pose)
    #     normalization["head"] = max(head)
    #     # normalization["head_min"] = min(head)
    #     # normalization["pose_min"] = min(pose)
    #     # normalization["head_mean"] = mean(head)
    #     # normalization["pose_mean"] = mean(pose)
    #     # normalization["head_mean"] = mean(head)
    #     # normalization["pose_mean"] = mean(pose)

    #     with open(NORM_JSON, 'w') as f:
    #         json.dump(normalization, f, indent=4)

    return dataset
