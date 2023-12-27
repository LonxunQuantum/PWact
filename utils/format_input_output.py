def make_iter_name(iter_index: int) :
    iter_format = "%04d"
    return "iter." + (iter_format % iter_index)

def make_train_name(model_index: int):
    train_name = "%03d"
    return "train."+(train_name % model_index)