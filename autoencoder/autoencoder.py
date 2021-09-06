from common.imports import *
from common.parameters import *
from module import get_data, create_model, model_train, model_test, tree_generator, sum_delta

##Clean
model_train.clean()
model_test.clean()
tree_generator.clean()

##Run
tree_generator.run()
img_w, img_h, train_set, test_set, train_cla_n, test_cla_n, train_img_n, r_train, r_test, r_max = get_data.run()

train_set, test_set, delta = sum_delta.run(act_name, img_w, img_h, train_set, test_set, r_train, r_max)

autoencoder, r, r_max = create_model.run(img_w, img_h, r_max, delta)
model_train.run(autoencoder, train_set, train_cla_n, train_img_n, r, r_train, r_max)
model_test.run(autoencoder, train_set, test_set, train_cla_n, test_cla_n, r_train, r_test, r_max, (img_w,img_h))
