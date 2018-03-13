import tensorflow as tf

def make_train_dict():
    train_dict = {}
    with open("train.vocab") as f:
        for line in f:
            (key, val) = line.split(',')
            train_dict[key] = val
    return train_dict

if __name__ == "__main__":
    print("starting")
    train_dict = make_train_dict()
    print(train_dict['mano'])
