from ml_collections import config_dict


def get_MPC_config():
    config = config_dict.ConfigDict()

    config.SYS_ID_DATA = 1

    config.PLANNING_HORIZON = 25

    config.INIT_VAR_DIVISOR = 4  # TODO what is this and where should it be?

    config.NUM_CANDIDATES = 25  # 30
    config.GAMMA = 1.25  # This is something else not discount value
    config.DISCOUNT_FACTOR = 1.0
    config.N_ELITES = 3  # 6  # TODO check it works for diff numbers of this
    config.iCEM_ITERS = 3
    config.XI = 0.3

    config.ACTIONS_PER_PLAN = 6  # 1  # How many actions to keep out of each total plan iteration, must be less than planning horizon

    config.BETA = 3.0  # 6.0

    config.ACQUISITION_SAMPLES = 15
    config.OPTIMISATION_ITERS = 5  # TODO think this shoud be diff from config.iCEM_ITERS but should check

    config.PRETRAIN_RESTARTS = 5
    config.TRAIN_GP_NUM_ITERS = 1000
    config.GP_LR = 0.01
    config.NUM_INDUCING_POINTS = 100

    config.ROLLOUT_SAMPLING = True

    """
    Current iCEM adaptions
    Use constant batch size
    Also use constant sample size for iCEM
    Check saved plan and best plan are the same thing, is there anyway the best plan may not be saved?
    """

    return config