from turntaking.dataload.dataset.scope import load_scope
from pprint import pprint

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        dset = load_scope(split=split)

    # dset = load_noxi(split="train")

    d = dset[0]  # type: ignore

    pprint(d)
