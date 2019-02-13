import numpy as np
import imageio
import os,time
import visual_words
import multiprocessing
import skimage.measure

def get_feature_one_image(args):
    i, image_path, label = args
    dictionary = np.load("dictionary.npy")
    num_SPM = 3
    dict_size = dictionary.shape[0]
    feature = np.reshape(get_image_feature(image_path, dictionary, num_SPM, dict_size), (1, -1))
    np.savez("../tmp/hist_features/"+"hist_"+str(i)+".npz", feature = feature, label = label)
    
    
def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    num_SPM = 3
    folderPath = "../data/"
    imagesNames = train_data['files']
    K = dictionary.shape[0]
    features = np.array([], dtype=np.int64).reshape(0,int(K*(4**num_SPM-1)/3))
    labels = train_data['labels']
    os.makedirs("../tmp/hist_features", exist_ok=True)
    with multiprocessing.Pool(num_workers) as pool:
         args = [(i, folderPath + imageName, labels[i]) for i, imageName in enumerate(imagesNames)]
         pool.map(get_feature_one_image, args)
    for i in range(train_data['files'].shape[0]):
        temp = np.load("../tmp/hist_features/"+"hist_"+str(i)+".npz")
        features = np.vstack([features, temp['feature']])
        labels = np.append(labels, temp['label'])
    np.savez("trained_system.npz", features=features, labels=labels, dictionary=dictionary, SPM_layer_num=num_SPM)


def evaluate_one_image(args):
    i = args
    # load training system
    trained_system = np.load("trained_system.npz")
    features = trained_system['features']
    labels = trained_system['labels']
    dictionary = trained_system['dictionary']
    SPM_layer_num = int(trained_system['SPM_layer_num'])
    # load test data
    test_data = np.load("../data/test_data.npz")
    imagePath = "../data/"+test_data['files'][i]
    image = imageio.imread(imagePath)
    # get the wordmap of test data and predict
    wordmap = visual_words.get_visual_words(image, dictionary)
    hist = get_feature_from_wordmap_SPM(wordmap, SPM_layer_num, dictionary.shape[0])
    similarity = distance_to_set(hist, features)
    predict_label = np.argmax(similarity)
    # save predict info
    np.save("../tmp/test/"+"predict_"+str(i)+".npy", labels[predict_label])


def evaluate_recognition_system(num_workers=4):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    test_data = np.load("../data/test_data.npz")

    num_test = test_data['files'].shape[0]
    os.makedirs("../tmp/test", exist_ok=True)
    # subprocess
    with multiprocessing.Pool(num_workers) as p:
        args = [(index) for index, imageName in enumerate(test_data['files'])]
        p.map(evaluate_one_image, args)
    # Begin evaluate
    conf = np.zeros((8, 8))
    for i in range(num_test):
        predict_label = np.load("../tmp/test/"+"predict_"+str(i)+".npy")
        # if test_data['labels'][i] != predict_label:
        #     print (predict_label, test_data['labels'][i], test_data['files'][i])
        conf[test_data['labels'][i], predict_label] += 1
    # Accuracy
    accuracy = np.trace(conf) / np.sum(conf)
    return conf, accuracy


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    image = imageio.imread(file_path)
    wordmap = visual_words.get_visual_words(image, dictionary)
    return get_feature_from_wordmap_SPM(wordmap, layer_num, K)


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    minimum = np.minimum(word_hist, histograms)
    similarity = np.sum(minimum, axis=1)
    return similarity


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    hist, _ = np.histogram(wordmap, bins=dict_size, range=(0, dict_size-1))
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    hist_all = []
    L = layer_num - 1
    for layer in range(layer_num):
        if layer == 0 or layer == 1:
            W_layer = 2**(-L)
        else:
            W_layer = 2**(layer-L-1)
        num_cells = 2**layer
        rows = np.array_split(wordmap, num_cells, axis=0)
        for x in rows:
            columns = np.array_split(x, num_cells, axis=1)
            for y in columns:
                hist = get_feature_from_wordmap(y, dict_size)
                hist_all = np.append(hist_all, W_layer * hist/wordmap.shape[0]*wordmap.shape[1])         
    return hist_all/np.sum(hist_all)
