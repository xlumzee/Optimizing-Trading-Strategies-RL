DATA_PATH = '/Users/amulya/Desktop/Capstone/DSCI-601-Amy/Data/FeatureEngineered/AKAM_feature_engineeredv2.csv'

AGENT_PARAMS = {
    'memory_size': 10000,
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'learning_rate': 0.0001,
    'batch_size': 128,
    'update_target_every': 1000
}

NUM_EPISODES = 500
INITIAL_BALANCE = 100000
