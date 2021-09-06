l_max = 20
k_max = 20
d_thres = 0

D = False
C = True   
R = True

dataset = "human"
#dataset = "robot"

ACTIONS = ["fist", "grasp", "poke"]

IM_SIZE = (480,848)

U0 = 424 #421
V0 = 240 #243
FX = 417 #433
FY = 377 #433

c_or_dc = "DC/" if D else "C/"
INIT_PATH = "resource/initial/"
DESIR_PATH = "resource/desired/"
TREE_PATH = "output/tree/"
MODEL_PATH = "resource/model/" + dataset + "/" + c_or_dc
SEQUENCE_PATH = "output/sequence/"
ANNOTATION_PATH = "resource/annotation/"


