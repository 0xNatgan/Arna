# Hyperparameters for NLP
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
NUM_EXPERTS = 8
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
ROUTING = 'hard'  # 'soft' or 'hard' or 'gumbel'
TOP_K = 4 # Only used if ROUTING is 'hard'
FREEZE_BERT = True  # Whether to freeze BERT layers during training
LOAD_BALANCE = False  # Whether to use load balancing loss
LOAD_BALANCE_COEF = 0.2  # Coefficient for load balancing loss