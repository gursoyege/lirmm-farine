from common.imports import *

def clean():
    shutil.rmtree(TREE_PATH)

def run():
    if not os.path.exists(SEQUENCE_PATH):
        os.makedirs(SEQUENCE_PATH)
    if not os.path.exists(TREE_PATH):
        os.makedirs(TREE_PATH) 

    r_matrix, r_max = r_generator(IM_SIZE, U0, V0, FX, FY)

    fI = glob.glob(INIT_PATH + "*")
    fD = glob.glob(DESIR_PATH + "*")

    initImg = cv2.imread(fI[0], cv2.IMREAD_GRAYSCALE)
    initImg = np.expand_dims(initImg, axis=-1)
    initImg = np.divide(initImg,255.0, dtype=np.float32)

    desImg = cv2.imread(fD[0], cv2.IMREAD_GRAYSCALE)
    desImg = np.expand_dims(desImg, axis=-1)
    desImg = np.divide(desImg,255.0, dtype=np.float32)

    rDesImg = img2rimg(desImg, r_matrix, r_max)
    
    min_prev = 0
    l = 0
    k = 0
    outputs = {}

    prev_act = ""
    while l<6:
        ds = []
        names = []  
        if l == 0:
            images = [fI[0]]
            path = TREE_PATH
        else:
            path = path + "*/"
            images = glob.glob(path + "*.png")

        for enum,img in enumerate(images):
            split = img.split('/')
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)
            img = np.divide(img,255.0, dtype=np.float32)
            img = img2rimg(img, r_matrix, r_max)

            for act_e, act in enumerate(ACTIONS):
                roi_size = np.load(MODEL_PATH + act + "/roi_dims.npy")
                delta, delta_res, mask, autoencoder = create_model(roi_size, r_max, MODEL_PATH, act, D, C)
                center = [0,0]
                centers = []
                annotations = glob.glob(ANNOTATION_PATH + act + "/*.xml")
                for ann in annotations:
                    action, name, loc_curr = getXml(ann)
                    center[1] = int(loc_curr[0] + (loc_curr[2] - loc_curr[0])/2)
                    center[0] = int(loc_curr[1] + (loc_curr[3] - loc_curr[1])/2)
                    centers.append(tuple(center.copy()))

                for cnt in centers:

                    save_path = TREE_PATH
                    for ind in range(l):
                        s = split[ind+2]
                        save_path = save_path + s + "/"

                    loc, roi = img2roi(act, img, IM_SIZE, cnt, roi_size)
                    
                    roi = np.expand_dims(roi, axis=0)

                    r_loc = get_r_loc(r_matrix, loc)
                    
                    pred = get_pred(roi, delta, delta_res, mask, autoencoder, r_loc, r_max, D, C, R)
                    
                    pred = pred[0,:,:,:]
                    
                    res = roi2rimg(loc, pred, img, r_matrix, r_max)
                    
                    d = dist(res, rDesImg, IM_SIZE, r_max)
                    ds.append(d)
                    prev_act = act
                    
                    if not os.path.exists(save_path + act + "_" + str(cnt[0]) + "_" + str(cnt[1]) + "/"):
                        os.makedirs(save_path + act + "_" + str(cnt[0]) + "_" + str(cnt[1]) + "/")
                    
                    res = res*r_max*255.0/r_matrix
                    cv2.imwrite(save_path + act + "_" + str(cnt[0]) + "_" + str(cnt[1]) + "/" + "img.png" ,res)
                    name_curr = (save_path + act + "_" + str(cnt[0]) + "_" + str(cnt[1])).split('/')[2:]
                    names.append(save_path + act + "_" + str(cnt[0]) + "_" + str(cnt[1]))

                    outputs[''.join(name_curr)] = d
                if ds != []:
                    ind = np.argmin(ds)
                    print(np.min(ds))
                    p = names[ind]
                    out_split = p.split('/')[2:]
                    out = ' + '.join(out_split)
                    print(out)
            if ds != [] and np.min(ds) < 0:
                ind = np.argmin(ds)
                s = names[ind]

            k += 1

        if l != 0 and np.min(ds) > min_prev:
            ds = min_prev
            out = min_prev_seq

            print(ds)
            print(out)
            cv2.imwrite(TREE_PATH + 'min.png', curr_min)
            break
        min_prev = np.min(ds)
        min_prev_seq = out

        curr_min = prune(ds, names, IM_SIZE, 1)
        
        l = l+1

    meth = '----'
    if D: meth += 'D'
    if C: meth += 'C'
    if R: meth += 'R'
    meth += ' \n'
    with open(SEQUENCE_PATH +"outputs.txt", "a") as text2:
        print(meth, file = text2)
        for x in outputs:
            print (x, outputs[x], file = text2)
        print (' \n', file = text2)

    meth += str(np.min(ds)) + ' \n'
    meth += out +' \n'
    with open(SEQUENCE_PATH +"sequence.txt", "a") as text:
        text.write(meth)

    cv2.imwrite(TREE_PATH + 'min.png', curr_min)


    
