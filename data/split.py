import json

def split(filename_prefix):
    data_folder = "/mnt/data/upcast/data/"
    with open(data_folder + filename_prefix + "_metadataset_restricted_values.json") as file:
        labels = json.load(file)

    # shuffle
    assert len(labels) == 11371 # should not change after we have started tuning
    items = list(labels.items())
    import random
    random.seed(1) # same each time!
    random.shuffle(items)

    # define train/test sets
    train_labels = dict(items[:5000])
    holdout_labels = dict(items[5000:]) #can increase the number after tuning, but not decrease

    with open(data_folder + filename_prefix + "_metadataset_train.json", "w") as file:
        json.dump(train_labels, file)
    with open(data_folder + filename_prefix + "_metadataset_holdout.json", "w") as file:
        json.dump(holdout_labels, file)


if __name__ == "__main__":
    1/0 # should not run this again!
    split("arxpr")
    split("arxpr2_25")
