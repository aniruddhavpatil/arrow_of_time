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


def disp_im(img):
	cv2.imshow("frame",img)
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
    
    global video_desc_dataset

    for i in range(len(kp)):

            (x, y) = kp[i].pt
            if kp[i].size < kp_sz_thresh:
            	continue
            interest_pt = (int(x+0.5), int(y+0.5))
            (c, r) = interest_pt
            ret_val, desc = create_descriptor(interest_pt, hsv[r-6:r+6, c-6:c+6, :])	# takes an avg_runtime of 0.001 sec
            
            return (ret_val, desc)

    # return frame_desc_dataset


def compute_optical_flow(prvs, next, hsv):
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return bgr, hsv

def compute_sift_kp(bgr):
    gray= cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY) 
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    kp_img=cv2.drawKeypoints(gray,kp, gray, flags=4)
    # disp_im(kp_img)
    return kp



def get_flow_words(frame1, frame2):

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
    (ret_val, desc) = get_descriptors_of_curr_frame(kp, hsv)	# takes avg_runtime of 0.003sec

    if(ret_val == 1):
	    if video_desc_dataset.shape == (1,0):
	        video_desc_dataset = np.hstack((video_desc_dataset, desc))
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


def build_hist_A(train_path):
	# load vocab of A
	np.load(vocab_of_A)

	# loop through all train videos
	fwd_play(train_path, )

		# get flow words for the video
		# apply knn to find the bin of flow words
		# store the histogram in hist_A

	# save hist_A





kp_sz_thresh = 15
no_of_clusters = 5
max_iter = 1000
no_of_rand_inits = 10

fwd_play("/home/goutham/CV_Project/test")
# need to take sqrt values to improve performance
build_vocab(no_of_clusters, max_iter, no_of_rand_inits)
np.save("fwd_vocab", vocab_of_A)




