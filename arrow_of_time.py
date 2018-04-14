import cv2
import os
import sys
import timeit
import numpy as np


dict_of_flow_words = np.array([[]])
bag_of_words = []
knn = cv2.ml.KNearest_create()

def disp_arr(bgr, r, c):
    for i in range(r):
        for j in range(c):
            print(bgr[i,j], end="   ")
        print("")


def disp_im(img, fn):
    cv2.imshow(fn,img)
    if cv2.waitKey(30) & 0xFF == 27:
        sys.exit()

def softmax(mag):
    mag_t = mag - np.mean(mag)
    return np.exp(mag_t)/np.sum(np.exp(mag_t))

def vectorize(ang):
    r,c = ang.shape
    temp = []
    for i in range(r):
        for j in range(c):
            temp.append(ang[i, j])
    return np.array(temp)



def create_descriptor(interest_pt, neighbourhood):
    # creates a softmax using magnitude values and find the
    # dominant direction of optical flow the region, return 16 (u,v)s => 32d-vector
    # could also make a similar sift descriptor but 16*8 => 128d-vector takes high
    # computation time for k means and k-nn

    ret_val = 1
    desc = []

    if neighbourhood.shape != (12, 12, 3):
        ret_val = 0
        return ret_val, desc

    for x in range(4):
        for y in range(4):

            mag = neighbourhood[(x*3):(x*3)+3, (y*3):(y*3)+3, 2]
            ang = neighbourhood[(x*3):(x*3)+3, (y*3):(y*3)+3, 0]
            mag = np.ravel(mag)#vectorize(mag)
            ang = np.ravel(ang)#vectorize(ang)
            sm = softmax(mag)
            theta = np.sum(np.multiply(sm, ang))
            max_mag = np.sum(np.multiply(sm, mag))
            theta = theta*(np.pi/180.0)
            vec = (max_mag*np.sin(theta), max_mag*np.cos(theta))
            desc.append(vec[0]); desc.append(vec[1]);

    # desc.shape => (1,32)
    desc = np.array([desc])
    return ret_val, desc


def get_descriptors_of_curr_frame(kp, hsv):

    kp_desc = np.array([[]])

    for i in range(len(kp)):

            (x, y) = kp[i].pt
            if kp[i].size < kp_sz_thresh:
                continue
            interest_pt = (int(x+0.5), int(y+0.5))
            (c, r) = interest_pt
            ret_val, desc = create_descriptor(interest_pt, hsv[r-6:r+6, c-6:c+6, :])    # takes an avg_runtime of 0.001 sec

            if(ret_val == 1):
                if kp_desc.shape == (1,0):
                    kp_desc = np.hstack((kp_desc, desc))
                else:
                    kp_desc = np.vstack((kp_desc, desc))


    return kp_desc

    # return frame_desc_dataset


def compute_optical_flow(prvs, next, hsv):
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # disp_im(bgr, "f1")
    return bgr, hsv

def compute_sift_kp(bgr):
    gray= cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    kp_img=cv2.drawKeypoints(gray,kp, gray, flags=4)
    # disp_im(kp_img, "f2")
    return kp



def get_flow_words_of_curr_frame(frame1, frame2):


    #  a total of 0.7 sec approx for getting flow words from a frame

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255


    # compute optical flow
    bgr, hsv = compute_optical_flow(prvs, next, hsv)    # takes avg_runtime of 0.3sec


    # compute sift interest points
    kp = compute_sift_kp(bgr)   # takes avg_runtime of 0.3sec


    # create descriptors
    desc = get_descriptors_of_curr_frame(kp, hsv)   # takes avg_runtime of 0.003sec


    # print(desc.shape)
    return desc


def kmeans_of_dict(no_of_clusters, max_iter, no_of_rand_inits):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0) # max_iter, acc
    ret,label,center=cv2.kmeans(np.float32(dict_of_flow_words),no_of_clusters,None,criteria,no_of_rand_inits,cv2.KMEANS_PP_CENTERS) # no_of_clusters, no_of_rand_inits
    print("MSE:", ret/dict_of_flow_words.shape[0])
    return center




def build_dict(abs_fname):

    global dict_of_flow_words

    imgs = sorted(os.listdir(abs_fname))

    for i in range(len(imgs)):

        if i+2 > len(imgs)-1:
            break

        prev = imgs[i]
        nxt = imgs[i+2]
        print(i)
        prvs = cv2.imread(abs_fname+"/"+prev)
        next = cv2.imread(abs_fname+"/"+nxt)

        desc = get_flow_words_of_curr_frame(prvs, next)

        if desc.shape != (1,0):
        # if there are no valid keypoint descriptors in current frame
            if dict_of_flow_words.shape == (1,0):
                dict_of_flow_words = np.hstack((dict_of_flow_words, desc[0,:].reshape(1,32)))
                dict_of_flow_words = np.vstack((dict_of_flow_words, desc[1:,:]))
            else:
                dict_of_flow_words = np.vstack((dict_of_flow_words, desc))

        print(dict_of_flow_words.shape)



def build_dict_of_flow_words(path):

    # pick random sample from the test data containing both back and fwd videos
    # making sure that equal no_of both fwd, backwd (how many ever pairs possible)
    # videos are represented while creating the dictionary

    # kmeans initialization
    no_of_clusters = 4000
    max_iter = 1000
    no_of_rand_inits = 10

    folder_names = os.listdir(path)
    thresh = 1000000  # no_of_examples


    fwd_lst = []
    bck_lst = []

    for subdir,dirs,files in os.walk(path):
            # build_dict_of_flow_words(path)
        for d in dirs:
            if d == 'test' or d == 'train' or d == 'validation':
                continue
            if 'train' in subdir:
                if d[0] == 'F':
                    fwd_lst.append(subdir + "/" + d)
                else:
                    bck_lst.append(subdir + "/" + d)

    f_l = len(fwd_lst)
    b_l = len(bck_lst)

    f_rand = np.arange(f_l)
    b_rand = np.arange(b_l)

    np.random.shuffle(f_rand)
    np.random.shuffle(b_rand)

    # print(f_rand, b_rand)

    for i in range(max(f_l, b_l)):


        if i<f_l:
            print(fwd_lst[f_rand[i]])
            build_dict(fwd_lst[f_rand[i]])
        if i<b_l:
            print(bck_lst[b_rand[i]])
            build_dict(bck_lst[b_rand[i]])

        if dict_of_flow_words.shape[0] >= thresh:
            break

    global bag_of_words
    bag_of_words = kmeans_of_dict(no_of_clusters, max_iter, no_of_rand_inits)
    np.save("dict_of_flow_words", dict_of_flow_words)
    np.save("bag_of_words", bag_of_words)


def build_knn():
    global bag_of_words

    n = bag_of_words.shape[0]
    indices = np.float32(np.array(range(0,n)))
    indices.reshape(n,1)

    knn.train(bag_of_words.astype(np.float32),cv2.ml.ROW_SAMPLE,indices.astype(np.float32))

def fwd_playABCD(path,label):
    hist_vid = np.zeros(4000)
    #path = '/home/aaron.v/ArrowSplit1/test/F_pZcB4oJ5jsc'
    print("Computing descriptors for ",path," video...")
    imgs = sorted(os.listdir(path))
    if label == 'C' or label == 'D':
        imgs = sorted(os.listdir(path),reverse=True)
    strt = timeit.default_timer()
    tr = 'train'
    if 'test' in path:
        tr = 'test'
    elif 'validation' in path:
        tr = 'validation'
    fname = path.split('/')[-1]
    # print('yo')
    # path_temp = '/home/aaron.v/arrow_of_time/hist_each_vid/%s/hist_%s_%s.npy' % (tr,label,fname)
    # if os.path.exists(path_temp):
    #     # print('exists')
    #     print('skip')
    #     return
    for i in range(len(imgs)):
        if i+2 > len(imgs)-1:
            break
        prev = imgs[i]
        nxt = imgs[i+2]
        print(i, end=", ")
        prvs = cv2.imread(path + "/" + prev)
        next = cv2.imread(path + "/" + nxt)
        if label == 'B' or label == 'D':
            prvs = cv2.flip(prvs, 1)
            next = cv2.flip(next, 1)
        desc = get_flow_words_of_curr_frame(prvs, next)
        desc = desc.astype(np.float32)
        if desc.shape[1] == 32:
            for i in desc:
                i = i.reshape(1,32)
                ret, results, neighbours, dist = knn.findNearest(i, 1)
                b = int(results[0][0])
                hist_vid[b] += 1
    print(hist_vid.shape)
    np.save('hist/train2/hist_%s_%s' % (label,fname), hist_vid)
    stp = timeit.default_timer()
    print("time taken: ", stp-strt, " secs")
    #return hist_vid

def kfold_run(path):
    hist_A_vid = []
    hist_B_vid = []
    hist_C_vid = []
    hist_D_vid = []
    # loop through the given dataset
    for subdir,dirs,files in os.walk(path):
            # build_dict_of_flow_words(path)
        for d in dirs:
            if d == 'test' or d == 'train' or d == 'validation':
                continue
            fwd_playABCD(subdir + "/" + d, 'A')
            fwd_playABCD(subdir + "/" + d, 'B')
            fwd_playABCD(subdir + "/" + d, 'C')
            fwd_playABCD(subdir + "/" + d, 'D')

    # np.save('hist_A', hist_A_vid)
    # np.save('hist_B', hist_B_vid)
    # np.save('hist_C', hist_C_vid)
    # np.save('hist_D', hist_D_vid)

    # hist_A_vid = np.load(f1)
    # hist_B_vid = np.load(f2)
    # hist_C_vid = np.load(f3)
    # hist_D_vid = np.load(f4)

kp_sz_thresh = 15
# build_dict_of_flow_words("/home/aaron.v/ArrowSplit1")   # give path to the train dataset, it builds the dict of flow words
dict_of_flow_words = np.load('dict_of_flow_words.npy', mmap_mode='r')
bag_of_words = np.load('bag_of_words.npy', mmap_mode='r')
build_knn()
kfold_run("/home/aaron.v/ArrowSplit1/train2")

# -------------------------------------------------------------------- Please read the below section

# # Note:

# # this is used to load the saved dict_of_flow_words array

# # pls follow the below steps while using knn
# # convert your desc to float32 (imp step)
# x= dict_of_flow_words[1000,:].astype(np.float32) #this is just an example
# # reshape it to the following size
# x=x.reshape(1,32)
# # use knn to find the bin
# ret, results, neighbours, dist = knn.findNearest(x, 1)
