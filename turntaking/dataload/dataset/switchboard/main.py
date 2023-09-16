from turntaking.dataload.dataset.switchboard import load_switchboard

if __name__ == "__main__":
    # for split in ["train", "val", "test"]:
    #    dset = load_switchboard(split=split)

    dset = load_switchboard(
        split="train",
        train_files="/ahc/work2/kazuyo-oni/turntaking/turntaking/dataload/dataset/switchboard/files/debug_train.txt",
    )
    dset = load_switchboard(split="train")

    d = dset[0]

    print(d)
