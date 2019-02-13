import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    if len(image.shape) == 2:
        image = np.tile(image[:, newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    filter_responses = extract_filter_responses(image)
    H = filter_responses.shape[0]
    W = filter_responses.shape[1]
    F = filter_responses.shape[2]
    filter_responses = filter_responses.reshape(H*W, F)
    dist = scipy.spatial.distance.cdist(filter_responses, dictionary)
    wordmap = np.argmin(dist, axis=1)
    wordmap = wordmap.reshape(H, W)
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    i,alpha,image_path = args
    image = skimage.io.imread(image_path)
    filter_responses = extract_filter_responses(image)
    x_axis = np.random.choice(filter_responses.shape[0], alpha)
    y_axis = np.random.choice(filter_responses.shape[1], alpha)
    response = filter_responses[x_axis, y_axis, :]
    if not os.path.exists("../tmp/train"):
        os.makedirs("../tmp/train")
    np.save("../tmp/train/"+ str(i) + ".npy", response)
    
    
def compute_dictionary(num_workers=4):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    train_data = np.load("../data/train_data.npz")
    # Load training data
    folderPath = "../data/"
    imagesNames = train_data['files']
    sensible_alpha = 200
    sensible_k = 100
    # subprocesses
    with multiprocessing.Pool(processes = num_workers) as pool:
        args = [(i, sensible_alpha, folderPath + imageName) for i, imageName in enumerate(imagesNames)]
        pool.map(compute_dictionary_one_image, args)
    filter_responses = None
    for i in range(imagesNames.shape[0]):
        if filter_responses is None:
            filter_responses = np.load("../tmp/train/"+ str(i) + ".npy")
        else:
            filter_responses = np.vstack((filter_responses, np.load("../tmp/train/"+ str(i) + ".npy")))
    # calculate K means
    kmeans = sklearn.cluster.KMeans(n_clusters=sensible_k).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save("dictionary.npy", dictionary)
    

