from turntaking.dataload.dataset.eald import load_eald
from pprint import pprint

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        dset = load_eald(split=split)

    # dset = load_noxi(split="train")

    d = dset[0]  # type: ignore

    pprint(d)
