from turntaking.dataload.dataset.switchboard import load_switchboard

if __name__ == "__main__":
    # for split in ["train", "val", "test"]:
    #    dset = load_switchboard(split=split)

    dset = load_switchboard(
        split="train",
        train_files="/ahc/work2/kazuyo-oni/conv_ssl/conv_ssl/conf/data_cut/train.txt",
    )
    # dset = load_switchboard(split="train")

    # d = dset[0]

    # pprint(d)
