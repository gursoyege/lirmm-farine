#train_data = "human"
train_data = "robot"

#test_data = "human"
test_data = "robot"

#act_name = "poke"
#act_name = "grasp"
act_name = "knock"

train_mean = True
test_mean = True
train_delta = True

c_or_dc = "DC/" if train_delta == True else "C/"

DATA_PATH = "resource/" + train_data + "/data/"  + act_name + "/"
R_PATH = "resource/" + train_data + "/r_value/"  + act_name + "/"

TEST_DATA_PATH = "resource/" + test_data + "/data/"  + act_name + "/"
TEST_R_PATH = "resource/" + test_data + "/r_value/"  + act_name + "/"

OUTPUT_PATH = "output/train/" + train_data + "/" + c_or_dc + act_name + "/"
TEST_OUT_PATH = "output/test/" + c_or_dc + train_data + "_on_" + test_data + "/"  + act_name + "/"

IM_SIZE = (480,848)
U0 = 424 #421
V0 = 240 #243
FX = 417 #433
FY = 377 #433

EPOCHS = 500
CLA_PASS = 1 # number of sample from each class

