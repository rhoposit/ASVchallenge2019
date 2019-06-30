from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')

import keras
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.utils import plot_model
from keras.datasets import mnist
from keras import regularizers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, MaxPooling1D, merge, LSTM, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, h5py, json
import numpy as np
np.set_printoptions(suppress=True)
from collections import defaultdict

###################################################################################
# To Run:
# python CNN_ablations.py runtype (i.e. the lead feature for ablations, see below)
# python CNN_ablations.py mfcc
###################################################################################

runtype = sys.argv[1]



coefs_ref = {"mfcc":70, "imfcc":60, "rfcc":30, "scmc":40, "mfccD":70, "imfccD":60, "rfccD":30, "scmcD":40, "mfccDD":70, "imfccDD":60, "rfccDD":30, "scmcDD":40, "cqcc":30, "cqccD":30, "cqccDD":30, "lfcc":70, "lfccD":70, "lfccDD":70, "xA":10, "xEA":10, "xE":10, "xEAs":10, "xEs":10,"xAs":10, "xS":10, "xSs":10}
pos = 1
neg = -1

task = "PA"
frames = 10
activation = "tanh"
loss = "mse"



# load up the data
def load_xvec(feat, data):
    infile = "xvecs/"+task+"_"+data+"_"+feat+".npy"
    fdict = np.load(infile)[()]
    X = np.array(list(fdict.values()))
    y_data = np.array(list(fdict.keys()))
    new_dict = defaultdict(list)
    for i in range(0, y_data.shape[0]):
        fname = y_data[i].split("_")[-1]
        new_dict[fname] = X[i]
    return new_dict



# load up the data
def load_h5data(feat, frame_bin, data):
    infile = task+"_"+data+"_resample/"+task+"_"+data+"_"+feat+"_bin"+str(frames)+".h5"
    coefs = coefs_ref[feat]
    f = h5py.File(infile,'r')
    data = f.get("Features")[:,:]
    y = json.loads(f.get("Targets")[()])
    X1 = data[:,:coefs*frames]
    data_dict = defaultdict(list)
    for i in range(0, len(y)):
        try:
            item = y[i]
            dat = X1[i]
            new = np.array(item.split(","))
            fname = new[0].split(".wav")[0][8:]
            data_dict[fname] = dat
            p = fname
            pd = dat
        except:
            data_dict[p] = pd
    return data_dict


# rescale training data values
def scale_neg_pos(data, domain):
    minval = domain[0]
    ptpval = domain[1]
    X_data = 2*(data - minval)/ptpval-1
    return X_data


def scale_zero_one(data, domain):
    out_range = (0,1)
    X_data = (data - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    X_data = X_data * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return X_data



def load_csv(infile):
    input = open(infile, "r")
    data = input.read().split("\n")[:-1]
    input.close()
    truth = defaultdict(int)
    attacks = defaultdict(str)
    spoofy = defaultdict(str)
    T, F, A, S = [], [], [], []
    for item in data:
        fname, envID, attackID, t = item.split(",")
        fname = fname.split(".wav")[0].split("_")[-1]
        if t == "spoof":
            truth[fname] = -1
            T.append(-1)
        if t == "bonafide":
            truth[fname] = 1
            T.append(1)
        attacks[fname] = attackID
        spoofy[fname] = t
        F.append(fname)
        A.append(attackID)
        S.append(t)
    return T, F, A, S


def dataprep(X1_train):
    domain1 = np.min(X1_train), np.ptp(X1_train)
    X1_train = scale_neg_pos(X1_train, domain1)
    X1_train = np.expand_dims(X1_train, axis=3)    
    return X1_train, domain1



def dataprep_test(X1_test,domain1):
    X1_test = scale_neg_pos(X1_test, domain1)
    X1_test = np.expand_dims(X1_test, axis=3)
    return X1_test

        

def generate_cm_file(frames, uttids, attackid, true_class, scores, std, lr, l2):
    std = str(std).replace(".", "_")
    lr = str(lr).replace(".", "_")
    l2 = str(l2).replace(".", "_")
    scores_output_file = task+"_scores/"+std+"."+lr+"."+l2+"."+abgroup+".txt"
    output = open(scores_output_file, "w")
    for i in range(0,len(uttids)):
        fname = task+"_D_"+uttids[i]
        attack = attackid[i]
        true = true_class[i]
        score = scores[i]
        outstring = fname+" "+attack+" "+true+" "+str(score)+"\n"
        output.write(outstring)
    output.close()
            


#########################################################################################
# define this DNN
class DNN():
    def __init__(self, total_cols, std, lr, l2_val, abgroup):
        self.rows = 1
        self.cols = total_cols
        self.channels = 1   #set to 3 if doing feat, featD, featDD
        self.num_classes = 2 # spoof and bonafide, one hotted
        self.shape = (self.cols, 1)
        self.std = std
        self.l2_val = l2_val
        if abgroup[0] == "x":
            self.k = 1
            self.m = 1
        else:
            self.k = 2
            self.m = 3

        
        optimizer = keras.optimizers.Adam(lr=lr)
        vec = Input(shape=self.shape)               
        self.classifier = self.build_classifier_CNN()
        labels = self.classifier(vec)

        self.classifier.compile(loss="mse",optimizer=optimizer,metrics=['mean_absolute_error'])        

    
    def build_classifier_CNN(self):
        k = self.k
        m = self.m
        model1 = Sequential()
#        model1.add(GaussianNoise(self.std, input_shape=self.shape))
        model1.add(BatchNormalization(input_shape=self.shape))
        model1.add(Conv1D(filters=32, kernel_size=k,input_shape=(self.shape),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val)))
        model1.add(MaxPooling1D(m))
        model1.add(Conv1D(filters=32, kernel_size=k,input_shape=(self.shape),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val)))
        model1.add(MaxPooling1D(m))
        model1.add(Conv1D(filters=32, kernel_size=k,input_shape=(self.shape),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val)))
        model1.add(MaxPooling1D(m))
        model1.add(Flatten())
        model1.add(Dense(1, activation='tanh'))
        model1.summary()
        vec = Input(shape=(self.shape))
        labels = model1(vec)
        return Model(vec, labels)


    def get_results(self, pred, truth, name):
        ref = ["spoof", "bonafide"]
        pred[pred>0] = 1
        pred[pred<=0] = 0
        truth[truth>0] = 1
        truth[truth<=0] = 0
        print(truth[0])
        print(pred[0])
        score = accuracy_score(truth, pred)
        # save the output
        outstring = "*********** "+name+" ***********\n"
        outstring += name+" - acc: "+str(100*score)+"\n"
        outstring += str(classification_report(truth, pred, target_names=ref))+"\n"
        outstring += str(confusion_matrix(truth, pred))+"\n"
        return outstring

    def plot_history(self, H, abgroup, std, lr, l2_val):
        # grab the history object dictionary
        H = H.history        
        # plot the training loss and accuracy
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H["loss"], label="train_loss")
        plt.plot(N, H["val_loss"], label="val_loss")
        plt.plot(N, H["mean_absolute_error"], label="train_mae")
        plt.plot(N, H["val_mean_absolute_error"], label="val_mae")
        plt.title(task+" CNN Training")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Error")
        plt.legend()
        # save the figure
        lr = str(lr).replace(".", "_")
        std = str(std).replace(".", "_")
        l2_val = str(l2_val).replace(".", "_")
        plt.savefig(task+"_plots/plot_adam."+abgroup+"."+std+"."+lr+"."+l2_val+".png")
        plt.close()        

        
#########################################################################################

def run_main(abgroup, epochs, std, lr, l2):
    global task, frames
    D = []
    Xtr = []
    Xt = []
    Ttrain, Ftrain, Atrain, Strain = load_csv(task+"_reference_train.csv")
    Ttest, Ftest, Atest, Stest = load_csv(task+"_reference_dev.csv")
    total_cols = 0
    F = abgroup.split("+")
    num_feats = len(abgroup.split("+"))
    for i in range(0, num_feats):
        feat = F[i]
        if feat[0] == "x":
            X1_train_recon, X1_test_recon = [], []
            X1_train_dict = load_xvec(feat, "train")
            X1_test_dict = load_xvec(feat, "dev")
            cols = 10
            total_cols += cols
            for i in range(0, len(Ftrain)):
                f = Ftrain[i]
                item = X1_train_dict[f]
                X1_train_recon.append(item)
            for i in range(0, len(Ftest)):
                f = Ftest[i]
                item = X1_test_dict[f]
                X1_test_recon.append(item)
            X1_train_recon = np.array(X1_train_recon)
            X1_test_recon = np.array(X1_test_recon)
            X1_train = np.expand_dims(np.array(X1_train_recon), axis=3)
            X1_test = np.expand_dims(np.array(X1_test_recon), axis=3)
            print(X1_train_recon.shape)
            print(X1_test_recon.shape)
            Xtr.append(X1_train)
            Xt.append(X1_test)
        else:
            X1_train_recon, X1_test_recon = [], []
            X1_train_dict = load_h5data(feat, frames, "train")
            X1_test_dict = load_h5data(feat, frames, "dev")
            coefs = coefs_ref[feat]
            cols = 10*coefs
            total_cols += cols
            for i in range(0, len(Ftrain)):
                f = Ftrain[i]
                item = X1_train_dict[f]
                X1_train_recon.append(item)
            for i in range(0, len(Ftest)):
                f = Ftest[i]
                item = X1_test_dict[f]
                X1_test_recon.append(item)

            X1_train_recon = np.array(X1_train_recon)
            X1_test_recon = np.array(X1_test_recon)
            X1_train, domain1 = dataprep(X1_train_recon)
            X1_test = dataprep_test(X1_test_recon, domain1)
            print(X1_train_recon.shape)
            print(X1_test_recon.shape)
            Xtr.append(X1_train)
            Xt.append(X1_test)
    
    X_train = np.concatenate(Xtr, axis=1)
    X_test = np.concatenate(Xt, axis=1)

    # initialize the DNN object
    dnn = DNN(total_cols, std, lr, l2, abgroup)
    val_method = "val_loss" 
    val_mode = "min" 
    batch_size = 32
    early_stopping = EarlyStopping(monitor=val_method,
                                   min_delta=0,
                                   patience=5,
                                   mode=val_mode)
    callbacks_list = [early_stopping]


    ############################# train the DNN on X1
    DNN1 = dnn.classifier.fit(X_train, Ttrain,
                              batch_size=batch_size,
                              validation_split=0.1,
                              epochs=epochs, shuffle=True,
                              callbacks=callbacks_list)


    # make predictions and gather a classification summary
    y_preds = dnn.classifier.predict(X_test).reshape(len(Ttest))
    generate_cm_file(frames, Ftest, Atest, Stest, y_preds, std, lr, l2)




if runtype == "singles":
    ABS = ["mfcc", "imfcc", "rfcc", "scmc", "lfcc", "cqcc", "xAs", "xEs", "xEAs", "xA", "xE", "xEA"]

if runtype == "mfcc":
    ABS = "mfcc+imfcc", "mfcc+rfcc", "mfcc+scmc", 

if runtype == "imfcc":
    ABS = ["imfcc+xAs", "imfcc+xEs", "imfcc+xEAs", "imfcc+xA", "imfcc+xE", "imfcc+xEA"]

if runtype == "rfcc":
    ABS = ["rfcc+scmc", "rfcc+lfcc", "rfcc+cqcc", "rfcc+xAs", "rfcc+xEs", "rfcc+xEAs", "rfcc+xA", "rfcc+xE", "rfcc+xEA"]

if runtype == "scmc":
    ABS = ["scmc+lfcc", "scmc+cqcc", "scmc+xAs", "scmc+xEs", "scmc+xEAs", "scmc+xA", "scmc+xE", "scmc+xEA"]

if runtype == "lfcc":
    ABS = ["lfcc+cqcc", "lfcc+xAs", "lfcc+xEs", "lfcc+xEAs", "lfcc+xA", "lfcc+xE", "lfcc+xEA"]
if runtype == "cqcc":
    ABS = ["cqcc+xAs", "cqcc+xEs", "cqcc+xEAs", "cqcc+xA", "cqcc+xE", "cqcc+xEA"]

ABS = ["scmc+xEAs"]



EPOCHS = [100]
noise_std = [0.00001, 0.0001, 0.001]
LEARN_RATES = [0.01, 0.001, 0.0001]
L2_VALS = [0.00001, 0.00005, 0.0001]



for abgroup in ABS:
    for epochs in EPOCHS:
        for std in noise_std:
            for lr in LEARN_RATES:
                for l2 in L2_VALS:
                    run_main(abgroup, epochs, std, lr, l2)


