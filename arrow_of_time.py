import cv2
import os
import sys
import timeit
import numpy as np

video_desc_dataset = np.array([[]])
vocab = 0

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
            ret_val, desc = create_descriptor(interest_pt, hsv[r-6:r+6, c-6:c+6, :])	# takes an avg_runtime of 0.001 sec

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
    disp_im(bgr, "f1")
    return bgr, hsv

def compute_sift_kp(bgr):
    gray= cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY) 
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    kp_img=cv2.drawKeypoints(gray,kp, gray, flags=4)
    disp_im(kp_img, "f2")
    return kp



def get_flow_words(frame1, frame2):

    global video_desc_dataset

	#  a total of 0.7 sec approx for getting flow words from a frame

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    
    # compute optical flow
    bgr, hsv = compute_optical_flow(prvs, next, hsv)	# takes avg_runtime of 0.3sec
    
    
    # compute sift interest points
    kp = compute_sift_kp(bgr)	# takes avg_runtime of 0.3sec
    
    
    # create descriptors
    desc = get_descriptors_of_curr_frame(kp, hsv)	# takes avg_runtime of 0.003sec
    # print(desc.shape)
    

    if desc.shape != (1,0):
        # if there are no valid keypoint descriptors in current frame

        if video_desc_dataset.shape == (1,0):
            video_desc_dataset = np.hstack((video_desc_dataset, desc[0,:].reshape(1,32)))
            video_desc_dataset = np.vstack((video_desc_dataset, desc[1:,:]))
        else:
            video_desc_dataset = np.vstack((video_desc_dataset, desc))
    
    
    print(video_desc_dataset.shape)
        


def build_vocab(no_of_clusters, max_iter, no_of_rand_inits):
	global vocab
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0) # max_iter, acc
	ret,label,center=cv2.kmeans(np.float32(video_desc_dataset),no_of_clusters,None,criteria,no_of_rand_inits,cv2.KMEANS_PP_CENTERS) # no_of_clusters, no_of_rand_inits
	vocab = center
	print("MSE:", ret/video_desc_dataset.shape[0])



def fwd_play(path):
	
	folder_names = sorted(os.listdir(path))

	for fname in folder_names:

		print("Computing descriptors for ",fname," video...")
		abs_fname = path+"/"+fname
		imgs = sorted(os.listdir(abs_fname))
		strt = timeit.default_timer()
		
		for i in range(len(imgs)):
			
			if i+2 > len(imgs)-1:
				break
		
			prev = imgs[i]
			nxt = imgs[i+2]
			print(i, end=", ")
			prvs = cv2.imread(abs_fname+"/"+prev)
			next = cv2.imread(abs_fname+"/"+nxt)

			get_flow_words(prvs, next)
    	
		stp = timeit.default_timer()
		print("time taken: ", stp-strt, " secs")

def build_dict_of_flow_words(path):

    # pick random sample from the test data
    # containing both back and fwd videos

    # for each video get desc and store it in
    # vocab[]. stop when you get 10^7 desc

    # perform kmeans on the vocab, and get
    # 4k cluster means

    # return cluster means



def get_hist_A_of_video(train_path):
    
    # load vocab of A
    np.load(vocab_of_A)

    # loop through all train videos
    fwd_play(train_path, )

        # get flow words for the video
        # apply knn to find the bin of flow words
        # store the histogram in hist_A

    # save hist_A



def kfold_run(path):

    # loop through the given dataset

        # build_dict_of_flow_words(path)

        # for each video in train_dataset

            # get_A_hist_of_video, store it in hist_A, lly generate label_A
            # get_B_hist_of_video, store it in hist_B, lly generate label_A
            # get_C_hist_of_video, store it in hist_C, lly generate label_A
            # get_D_hist_of_video, store it in hist_D, lly generate label_A

        # save the dataset
        # perform pca on each of the hists individually

        # train the dataset on MLP, to give A,B,C,D values given an input of all hist combined together

        # repeat the above steps with the test data except, instead of training on the model predict

    # reset the weights, vocab








kp_sz_thresh = 15
no_of_clusters = 5
max_iter = 1000
no_of_rand_inits = 10

fwd_play("/home/goutham/CV_Project/test")
# need to take sqrt values to improve performance
build_vocab(no_of_clusters, max_iter, no_of_rand_inits)
np.save("fwd_vocab", vocab_of_A)




