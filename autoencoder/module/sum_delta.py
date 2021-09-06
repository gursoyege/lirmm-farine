from common.imports import *

def clean():
    if os.path.exists(OUTPUT_PATH):
        if os.path.isfile(OUTPUT_PATH + "delta.npy"):
            os.remove(OUTPUT_PATH + "delta.npy")
        if os.path.isfile(OUTPUT_PATH + "beg_avg.png"):
            os.remove(OUTPUT_PATH + "beg_avg.png")
        if os.path.isfile(OUTPUT_PATH + "end_avg.png"):
            os.remove(OUTPUT_PATH + "end_avg.png")
        if os.path.isfile(OUTPUT_PATH + "delta.png"):
            os.remove(OUTPUT_PATH + "delta.png")

def run(act_name, img_w, img_h, train_set, test_set, r_train, r_max):
    beg_train = train_set[0:1,:,:,:,:,:]
    end_train = train_set[1:2,:,:,:,:,:]

    r_train_sum = np.sum(r_train, axis=(0), keepdims=True)

    beg_rsum = np.sum(beg_train, axis=(1,2), keepdims=True)
    end_rsum = np.sum(end_train, axis=(1,2), keepdims=True)
    
    r_train_sum = np.expand_dims(r_train_sum, axis=0)
    r_train_sum = np.expand_dims(r_train_sum, axis=0)


    beg_avg = beg_rsum/((beg_train).shape[1]*(beg_train).shape[2])
    end_avg = end_rsum/((end_train).shape[1]*(end_train).shape[2])
    r_avg = r_train_sum/(r_train).shape[0]

    delta = (beg_avg - end_avg)*r_avg/r_max

    np.save(OUTPUT_PATH + "delta.npy", delta[0,0,0,:,:,:])

    cv2.imwrite(OUTPUT_PATH + "beg_avg.png", (255 * (beg_avg[0,0,0,:,:,:])).astype(np.uint8))
    cv2.imwrite(OUTPUT_PATH + "end_avg.png", (255 * (end_avg[0,0,0,:,:,:])).astype(np.uint8))
    cv2.imwrite(OUTPUT_PATH + "delta.png", (255 * (delta[0,0,0,:,:,:])).astype(np.uint8))

    if train_delta == False:
        delta = np.zeros((1,1,1,img_w, img_h,1))

    train_set[0:1,:,:,:,:,:] = train_set[0:1,:,:,:,:,:] - delta[0:1,:,:,:,:,:]
    test_set[0:1,:,:,:,:,:] = test_set[0:1,:,:,:,:,:] - delta[0:1,:,:,:,:,:]
    return train_set, test_set, delta
