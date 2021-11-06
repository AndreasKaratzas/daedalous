
import numpy


CITY_ENV_TARGETS = numpy.array([
    [ 110, -270, -20],
    [ 183, -157, -27],
    [-219, - 97, -22],
    [ 150,   50, -35],
    [ 222, - 64, -27],
    [- 19, -232, -37],
    [ 272, - 29, -12],
    [ 170, -235, -86],
    [- 46, - 94, -39],
    [- 31, -115, -31],
    [ 103,   38, -26],
    [ 175,   64, -51],
    [ 102,  132, -42],
    [ 120,  250, -25],
    [ 214, -105, -31],
    [ 267, -180, -43]
], dtype=numpy.float)

AIRSIM_NH_ENV_TARGETS = numpy.array([
    [ 150, -160, - 2],
    [ 105,  215, -28],
    [-140, -126, - 2],
    [- 65,  110, -40],
    [ 160, - 90, - 6],
    [ 150,   31, -15],
    [ 120, - 84, - 6],
    [  54, - 15, -17],
    [ 125, - 15, -21],
    [  10, - 95, - 7],
    [- 40, -150, -50],
    [  95,  125, -40],
    [  75,    5, -12],
    [ 125,   35, -11],
    [- 50,   60, - 6],
    [- 40,  130, -10]
], dtype=numpy.float)

MEM_CAPACITY = int(1e5)
BATCH_SIZE = 128
TARGET_SYNC = int(1e3)
NUM_OF_STEPS_TO_CHECKPOINT = int(1e4)
EPISODES = int(5e3)
GAMMA = 0.995
ALPHA = 0.4
BETA = 0.6
PRIOR_EPS = 1e-6
V_MIN = -200.0
V_MAX = 200.0
N_ATOMS = 51
N_STEP = 3
VERBOSITY = 10
CLOCKSPEED = 1000
TARGET_DISTANCE_THRESHOLD = 5.0
MULTIROTOR_STEP = 1
DEPTH_CAMERA_QUANTIZATION = 36
LEARNING_RATE = 1e-4
SEED = 33
