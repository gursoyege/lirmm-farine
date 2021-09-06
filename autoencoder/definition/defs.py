import glob
import numpy as np
import tensorflow as tf
import keras

from numpy.random import default_rng
from keras import backend as K

def get_data(train_path, test_path, train_mean, test_mean):
    beginning = 1
    actionTrainFolder = sorted(glob.glob(train_path + "train/*/"))
    for ins_e, ins in enumerate(actionTrainFolder):
        instanceFolder = sorted(glob.glob(ins + "*/"))
        for cla_e, cla in enumerate(instanceFolder):
            classFolder = sorted(glob.glob(cla + "/*.png"))
            for img_e, img in enumerate(classFolder):
                tensor = tf.io.read_file(img)
                print(img)
                tensor = tf.io.decode_image(tensor, dtype=tf.dtypes.uint8)
                tensor = tf.image.convert_image_dtype(tensor, tf.float32)
                if beginning == 1:
                    img_w,img_h,img_d = tensor.shape
                    train_cla_n = len(instanceFolder)
                    train_img_n = len(classFolder)
                    train_set = np.zeros((len(actionTrainFolder) ,len(instanceFolder) ,len(classFolder), img_w, img_h, 1), dtype = 'float32')               
                    beginning = 0
                train_set[ins_e,cla_e,img_e,:,:,:] = tensor
    beginning = 1
    actionTestFolder = sorted(glob.glob(test_path + "test/*/"))
    for ins_e, ins in enumerate(actionTestFolder):
        instanceFolder = sorted(glob.glob(ins + "*/"))
        for cla_e, cla in enumerate(instanceFolder):
            classFolder = sorted(glob.glob(cla + "/*.png"))
            for img_e, img in enumerate(classFolder):
                tensor = tf.io.read_file(img)
                print(img)
                tensor = tf.io.decode_image(tensor, dtype=tf.dtypes.uint8)
                tensor = tf.image.convert_image_dtype(tensor, tf.float32)
                if beginning == 1:
                    test_cla_n = len(instanceFolder)
                    test_set = np.zeros((len(actionTestFolder) ,len(instanceFolder) ,len(classFolder), img_w, img_h, 1), dtype = 'float32')               
                    beginning = 0
                test_set[ins_e,cla_e,img_e,:,:,:] = tensor

    if train_mean == True:
        train_set = np.mean(train_set, axis=(2), keepdims=True)
    if test_mean == True:
        test_set = np.mean(test_set, axis=(2), keepdims=True)
    return img_w, img_h, train_cla_n, test_cla_n, train_img_n, train_set, test_set


def get_r_val(train_path, test_path, train_cla_n, test_cla_n):
    r_train = np.array([])
    r_test = np.array([])
    
    beginning = 1
    actionFolder = sorted(glob.glob(train_path + "train/*/"))
    
    for ins_e, ins in enumerate(actionFolder):
        print(ins)
        instanceFolder = sorted(glob.glob(ins + "*.npy"))
        for r_e, r in enumerate(instanceFolder):
            print(r)
            r_act = np.load(r)
            if beginning == 1:
                r_shape = r_act.shape
                r_train = np.zeros((train_cla_n, r_shape[0], r_shape[1], 1))
                r_train[r_e,:,:,:] = r_act
                beginning = 0
            else :
                r_train[r_e,:,:,:] = r_act
    
    beginning = 1
    actionFolder = sorted(glob.glob(test_path + "test/*/"))
    
    for ins_e, ins in enumerate(actionFolder):
        print(ins)
        instanceFolder = sorted(glob.glob(ins + "*.npy"))
        for r_e, r in enumerate(instanceFolder):
            print(r)
            r_act = np.load(r)
            if beginning == 1:
                r_shape = r_act.shape
                r_test = np.zeros((test_cla_n, r_shape[0], r_shape[1], 1))
                r_test[r_e,:,:,:] = r_act
                beginning = 0
            else :
                r_test[r_e,:,:,:] = r_act
    return r_train, r_test

def make_epoch(set_complete):
    ins_e,cla_e,img_e,img_h,img_w,img_d=set_complete.shape
    cla_rand = np.zeros(cla_e)
    img_rand = np.zeros(img_e)
    rng = default_rng()
    
    cla_rand[:] = rng.choice(cla_e, size=cla_e, replace=False)
    img_rand[:] = rng.choice(img_e, size=img_e, replace=False)


    cla_rang = np.tile(cla_rand,img_e)
    img_rang = np.repeat(img_rand, cla_e)

    cla_iter = iter(cla_rang)
    img_iter = iter(img_rang)
    return cla_iter, img_iter , cla_rang, img_rang


def make_batch(set_complete, cla_iter, img_iter, batch_size, r_train, r_max):
    ins_e,cla_e,img_e,img_h,img_w,img_d=set_complete.shape
    r_batch = np.zeros((batch_size,img_h,img_w,img_d),dtype='float32')
    input_batch = np.zeros((batch_size,img_h,img_w,img_d),dtype='float32')
    ground_batch = np.zeros((batch_size,img_h,img_w,img_d),dtype='float32')
    for i in range(batch_size):
        cla_pick = next(cla_iter, None)
        img_pick = next(img_iter, None)
        if cla_pick == None or img_pick == None:
            break
        r_batch[i,:,:,:] = r_train[int(cla_pick), :,:,:]
        input_batch[i,:,:,:] = set_complete[0,int(cla_pick),int(img_pick),:,:,:]        
        ground_batch[i,:,:,:] = set_complete[1,int(cla_pick),int(img_pick),:,:,:]
    input_batch = input_batch * r_batch/r_max
    ground_batch = ground_batch * r_batch/r_max

    return input_batch, ground_batch, r_batch


def custom_loss(r, img_w, img_h, r_max):
    img_w = K.constant(img_w)
    img_h = K.constant(img_h)
    r_max = K.constant(r_max)
    def loss(y_true, y_pred):
        loss_value = K.sum((K.abs(y_pred - y_true)))/(img_w * img_h)
        return loss_value
    return loss

def dist(a, b, IM_SIZE, r_max):
    z_diff = 50 #mm
    return np.sum(np.abs(a-b) * r_max * z_diff /(IM_SIZE[0] * IM_SIZE[1]))
