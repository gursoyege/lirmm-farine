from re import A
import numpy as np
import cv2
import glob
import os
import shutil
import tensorflow as tf
import keras
import lxml.etree as ET

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose, Reshape, LeakyReLU
from keras import backend as K
from tensorflow.python.ops.gen_math_ops import NotEqual

def getXml(filepath):
    tree = ET.parse(filepath)

    action = tree.findall('./action')[0].text

    name = tree.findall('./name')[0].text

    xmin = int(tree.findall('./bbox/xmin')[0].text)
    ymin = int(tree.findall('./bbox/ymin')[0].text)
    xmax = int(tree.findall('./bbox/xmax')[0].text)
    ymax = int(tree.findall('./bbox/ymax')[0].text)

    return (action, name, [xmin, ymin, xmax, ymax])

def r_generator(IM_SIZE, U0, V0, FX, FY):
    r_max = np.sqrt(1 + (U0/FX)**2 + (V0/FY)**2)
    r_matrix = np.zeros((IM_SIZE[0],IM_SIZE[1], 1))
    for y in range(IM_SIZE[0]):
        for x in range(IM_SIZE[1]):
            r_matrix[y,x,0] = np.sqrt(1 + ((x-U0)/FX)**2 + ((y-V0)/FY)**2)
    return r_matrix, r_max

def dist(a, b, IM_SIZE, r_max):
    z_diff = 50 #mm
    return np.sum(np.abs(a-b) * r_max * z_diff /(IM_SIZE[0] * IM_SIZE[1]))


def roi2rimg(loc, pred, image, r_matrix, r_max):
    image_temp =  image.copy()
    pred_blend = pred.copy()

    orig_roi = image_temp[int(loc[0]):int(loc[2]),int(loc[1]):int(loc[3]),:]
    image_temp[int(loc[0]):int(loc[2]),int(loc[1]):int(loc[3]),:] = pred_blend

    return image_temp

def img2roi(act, image, IM_SIZE, center, roi_size):

    loc = np.zeros([4])
    loc[0] = center[0] - roi_size[0]/2 #ymin
    loc[2] = center[0] + roi_size[0]/2 #ymax

    loc[1] = center[1] - roi_size[1]/2 #xmin
    loc[3] = center[1] + roi_size[1]/2 #xmax
    
    while loc[0]<0:
        loc[0] += 1
        loc[2] += 1
    while loc[1]<0:
        loc[1] += 1
        loc[3] += 1
    while loc[2]>IM_SIZE[0]:
        loc[0] -= 1
        loc[2] -= 1
    while loc[3]>IM_SIZE[1]:
        loc[1] -= 1
        loc[3] -= 1
    
    roi = np.zeros([roi_size[0],roi_size[1]])
    roi = image[int(loc[0]):int(loc[2]),int(loc[1]):int(loc[3]),:]
    return loc, roi

def img2rimg(image, r, r_max):
    rimg = r * image / r_max
    return rimg

def get_r_loc(r_matrix, loc):
    r_loc = r_matrix[int(loc[0]):int(loc[2]),int(loc[1]):int(loc[3]),:]
    return r_loc

def create_model(roi_size, r_max, MODEL_PATH, ACTION, D, C):
    img_w = roi_size[0]
    img_h = roi_size[1]

    weights = MODEL_PATH + ACTION + "/weights.h5"
    
    delta = np.load(MODEL_PATH + ACTION + "/delta.npy")
    delta_res = delta.copy()
    mask = get_mask(np.abs(delta), img_w*img_h)

    delta = np.expand_dims(delta, axis=0)

    if D == 0:
        delta = np.zeros((1,img_w, img_h,1)) #uncomment for Cxx
    if C == 1:
        autoencoder = get_model(False, img_w, img_h, delta, weights)
    else:
        autoencoder = None
    return delta, delta_res, mask, autoencoder

def get_pred(pre, delta, delta_res, mask, autoencoder, r_values, r_max, D, C, R):
    inp = pre - delta
    inp[inp > 1] = 1
    inp[inp < 0] = 0

    if C == 1:
        pred = autoencoder.predict(inp)
    else:
        pred = inp
    if R == 1:
        pred = refinement(pre, pred.astype('f'),mask, r_values, delta_res, r_max)

    #pred[pred > 1] = 1
    #pred[pred < 0] = 0
    return pred

def get_mask(delta, area):
    ret, mask = cv2.threshold((np.abs(delta)*255).astype('uint8'),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    knel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + int(np.floor(area/3000)), 1 + int(np.floor(area/3000))))
    mask = cv2.dilate(mask,knel, iterations = 1)

    mask = (mask/255).astype('f')
    mask = np.expand_dims(mask, axis=(0,-1))
    return mask

def get_model(if_D, img_w, img_h, delta, weights):
    input_img = Input(shape=(img_w, img_h, 1),name='input')
    x = Conv2D(32, (5, 5), padding='same', name='conv1')(input_img)
    x = LeakyReLU(alpha=0.2, name='leaky1')(x)
    x = MaxPooling2D((2, 2), padding='same', name='pool1')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = LeakyReLU(alpha=0.2, name='leaky2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv3')(x)
    x = LeakyReLU(alpha=0.2, name='leaky3')(x)
    x = MaxPooling2D((2, 2), padding='same', name='pool3')(x)

    shape = x.shape
    x = Flatten(name='flatten_conv2fc')(x)

    x = Dense(2048, name='fc_encoder')(x)
    x = LeakyReLU(alpha=0.2, name='leaky4')(x)

    x = Dense(shape[1]*shape[2]*shape[3], name='map_fc2deconv')(x)
    x = LeakyReLU(alpha=0.2, name='leaky5')(x)
    x = Reshape((shape[1],shape[2],shape[3]), name='reshape_fc2deconv')(x)

    x = UpSampling2D((2, 2), name='unpool1')(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv1')(x)
    x = LeakyReLU(alpha=0.2, name='leaky6')(x)
    x = UpSampling2D((2, 2), name='unpool2')(x)
    x = Conv2DTranspose(32, (3, 3), padding='same', name='deconv2')(x)
    x = LeakyReLU(alpha=0.2, name='leaky7')(x)
    x = UpSampling2D((2, 2), name='unpool3')(x)
    x = Conv2DTranspose(1, (5, 5), padding='same', name='deconv3')(x)
    x = LeakyReLU(alpha=0.2, name='leaky8')(x)

    unpad_h = np.array([np.ceil((x.shape[1]-img_w)/2), x.shape[1] - np.floor((x.shape[1]-img_w)/2)])
    unpad_v = np.array([np.ceil((x.shape[2]-img_h)/2), x.shape[2] - np.floor((x.shape[2]-img_h)/2)])

    decoded = keras.layers.Lambda(lambda x: x[:,int(unpad_h[0]):int(unpad_h[1]),int(unpad_v[0]):int(unpad_v[1]),:], name='reshape_out')(x)

    autoencoder = Model(input_img, decoded, name='autoencoder')
    autoencoder.load_weights(weights)
    return autoencoder

def refinement(pre_tmp,pred_tmp,mask,r_values,delta_res,r_max):

    pre =  pre_tmp[0,:,:,:].copy()
    pred = pred_tmp[0,:,:,:].copy()
    pred_blend = (pred_tmp[0,:,:,:].copy())*0

    m_pre_d = np.sum(pre*mask)/(mask>0).sum()#
    m_post_d = np.sum(pred*mask)/(mask>0).sum()#

    m_delta_d = np.sum(delta_res*mask)/(mask>0).sum()

    pred_blend = (pre - delta_res) * np.logical_not(mask[0,:,:,:]) + (pred + m_pre_d - m_post_d - m_delta_d) * mask[0,:,:,:]
    pred_blend = np.expand_dims(pred_blend, axis=0)

    return pred_blend

def prune(ds, names, IM_SIZE, top):
    tmp_imgs = np.zeros((top, IM_SIZE[0], IM_SIZE[1]))
    argmins = []
    if isinstance(ds, list):
        mins = sorted(set(ds))[:top]
    else:
        mins = [sorted(set([ds]))[:top]]
    for i, val in enumerate(mins):
        ind = np.argwhere(ds == val)[0,0]
        argmins.append(ind)
        tmp_imgs[i] = cv2.imread(names[ind] + '/img.png', cv2.IMREAD_GRAYSCALE)
        if i == 0:
            curr_min = cv2.imread(names[ind] + '/img.png', cv2.IMREAD_GRAYSCALE)

    files = glob.glob("output/tree/*/")
    for fs in files:
        shutil.rmtree(fs)
    
    for i, ind in enumerate(argmins):
        os.makedirs(names[ind])
        cv2.imwrite(names[ind] + '/img.png', tmp_imgs[i])
    return curr_min
