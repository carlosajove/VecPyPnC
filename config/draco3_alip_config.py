import torch

class SimConfig(object):
    CONTROLLER_DT = 0.01
    N_SUBSTEP = 10
    CAMERA_DT = 0.001
    KP = 0.
    KD = 0.

    INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.5 - 0.757]
    INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

    PRINT_TIME = False
    PRINT_ROBOT_INFO = False
    VIDEO_RECORD = False
    RECORD_FREQ = 1
    SIMULATE_CAMERA = False
    SAVE_CAMERA_DATA = False


class PnCConfig(object):
    DYN_LIB = "pinocchio"  # "dart"
    CONTROLLER_DT = SimConfig.CONTROLLER_DT
    SAVE_DATA = True
    SAVE_FREQ = 1

    PRINT_ROBOT_INFO = SimConfig.PRINT_ROBOT_INFO


class WBCConfig(object):
    VERBOSE = False

    # Max normal force per contact
    RF_Z_MAX = 1000.0

    # Task Hierarchy Weights
    W_COM = 100.0
    W_TORSO = 60.0
    W_UPPER_BODY = 40.0
    W_CONTACT_FOOT = 60.0
    W_SWING_FOOT = 80.0

    # Task Gains
    KP_COM = torch.tensor([1000., 1000., 1000])
    KD_COM = torch.tensor([50., 50., 50.])

    KP_TORSO = torch.tensor([100., 100., 100])
    KD_TORSO = torch.tensor([10., 10., 10.])

    # ['neck_pitch', 'l_shoulder_fe', 'l_shoulder_aa', 'l_shoulder_ie',
    # 'l_elbow_fe', 'l_wrist_ps', 'l_wrist_pitch', 'r_shoulder_fe',
    # 'r_shoulder_aa', 'r_shoulder_ie', 'r_elbow_fe', 'r_wrist_ps',
    # 'r_wrist_pitch'
    # ]
    KP_UPPER_BODY = torch.tensor([
        40., 100., 100., 100., 50., 40., 50., 100., 100., 100., 50., 40., 50.
    ])
    KD_UPPER_BODY = torch.tensor(
        [2., 8., 8., 8., 3., 2., 3., 8., 8., 8., 3., 2., 3.])

    KP_FOOT_POS = torch.tensor([2000., 2000., 1000.])
    KD_FOOT_POS = torch.tensor([100., 100., 30.])
    KP_FOOT_ORI = torch.tensor([1000., 1000., 1000.])
    KD_FOOT_ORI = torch.tensor([30., 30., 30.])

    # Regularization terms
    LAMBDA_Q_DDOT = 1e-8
    LAMBDA_RF = 1e-7

    # B_TRQ_LIMIT = True
    B_TRQ_LIMIT = False

    # Integration Parameters
    VEL_CUTOFF_FREQ = 2.0  #Hz
    POS_CUTOFF_FREQ = 1.0  #Hz
    MAX_POS_ERR = 0.2  #Radians



class WalkingState(object):
    STAND = 0
    BALANCE = 1
    RF_CONTACT_TRANS_START = 2
    RF_CONTACT_TRANS_END = 3
    RF_SWING = 4
    LF_CONTACT_TRANS_START = 5
    LF_CONTACT_TRANS_END = 6
    LF_SWING = 7
    SWAYING = 10
    ALIP = 19

class AlipParams(object):
    N_BATCH = 5
    TS = 0.2
    NT_qp = 4
    NT_mpc = 1
    NS = 4
    MASS = 35.
    ZH = 0.69
    WIDTH = 0.15
    G = 9.81
    UFP_X_MAX = 0.5
    UFP_Y_MAX = 0.4
    UFP_Y_MIN = 0.08
    LX_OFFSET = 0.
    LY_DES = 0.
    COM_YAW = 0.
    INITIAL_STANCE_LEG = -1
    SWING_HEIGHT = 0.05

    PX = 1.
    PY = 1.
    PLX = 1.
    PLY = 1.
    PBOUND = 5.

    """
    COM
    """
    UCOM_X_MAX = 0.3
    UCOM_Y_EXT = 0.3
    UCOM_Y_INT = 0.1

    """
    Mixed
    """
    ALPHA = 0.7

    RF_Z_MAX = 1000.0
