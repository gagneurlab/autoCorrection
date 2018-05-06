import os

DEFAULT_LEARNING_RATE = 0.00068
DEFAULT_ENCODING_DIM = 23
DEFAULT_BATCH_SIZE = None
DEFAULT_EPOCHS = 250

MODEL_PATH = os.path.join(os.getcwd(), "saved_models")
OPT_PARAM_PATH = os.path.join(MODEL_PATH, "best")

