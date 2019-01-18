# Variables used to load the saved model. Not all variables are strictly necessary

# Data processing save parameters
PICKLE_FOLDER_NAME = "fnc_fever_3"
PICKLE_SAVE_FOLDER = "pickle_data/" + PICKLE_FOLDER_NAME + "/"

# Checkpoints for 4 label fnc testing
RELATED_UNRELATED_MODEL_NAME = "fnc_related_unrelated_tf"
RELATED_UNRELATED_MODEL_FOLDER = "models/" + RELATED_UNRELATED_MODEL_NAME + "/"
RELATED_UNRELATED_MODEL_PATH = RELATED_UNRELATED_MODEL_FOLDER + RELATED_UNRELATED_MODEL_NAME

THREE_LABEL_MODEL_NAME = "fnc_fever_cnn_tf_dann_3_no_2"
THREE_LABEL_MODEL_FOLDER = "models/" + THREE_LABEL_MODEL_NAME + "/"
THREE_LABEL_MODEL_PATH = THREE_LABEL_MODEL_FOLDER + THREE_LABEL_MODEL_NAME

# Domain usage
USE_DOMAINS = True

# One of these must be selected as the primary input for training
# but multiple may be selected when processing data
USE_TF_VECTORS = False
USE_CNN_FEATURES = True

# Adds TF vectors to features before label prediction
ADD_FEATURES_TO_LABEL_PRED = True

# Model Size Parameters
BATCH_SIZE = 100
CNN_HEADLINE_LENGTH = 50
CNN_BODY_LENGTH = 500

# LABEL MAPPINGS
FNC_LABELS_REV = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}

