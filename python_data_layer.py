import numpy as np
import sys
import cv2
import caffe
import os
import random
import time
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

datadir = 'data/CelebA/'
mean_file = 'model/resnet_50/ResNet_mean.binaryproto'
proto_data = open(mean_file, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]
selected_attr = np.zeros(40)
for attr_index in range(40):
    selected_attr[attr_index] = attr_index


def pre_process(color_img):
    resized_img = cv2.resize(color_img, (224, 224))
    return np.transpose(resized_img, (2, 0, 1)) - mean


def showImage(img,points=None, bbox=None):
    if points is not None:
        for i in range(0,points.shape[0]/2):
            cv2.circle(img,(int(round(points[i*2])),int(points[i*2+1])),1,(0,0,255),2)
    if bbox is not None:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), (0,0,255), 2)
    plt.figure()
    plt.imshow(img)
    plt.show()
    print 'here'


class ValLayer(caffe.Layer):

    attri_num = 40

    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """
    total_namelist = []
    attri_array =[]
    # landmark_array =[]
    img_num = 0

    target_height = 224
    target_witdh = 224
    # mean = []
    batch = 20
    imgset = 'align_val'


    def setup(self, bottom, top):
        filename = os.path.join(datadir, self.imgset, self.imgset + 'list.txt')
        attrfile = os.path.join(datadir, self.imgset, self.imgset + '_attr.txt')
        det_df = read_csv(filename,sep=' ',header=None)
        self.total_namelist = det_df[det_df.columns[0]].values
        self.img_num = len(self.total_namelist)
        self.attri_array = np.zeros((self.img_num, self.attri_num))
        attr_f = open(attrfile, 'r')
        attr_line = attr_f.readline().strip().split()
        i = 0
        while attr_line:
            for j in range(0, self.attri_num):
                value = int(float(attr_line[j + 1]))
                assert value == 1 or value == -1
                self.attri_array[i, j] = value
            attr_line = attr_f.readline().strip().split()
            i += 1
        top[0].reshape(self.batch,3,self.target_height, self.target_witdh)
        top[1].reshape(self.batch, len(selected_attr))

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for i in range(self.batch):
            # print i
            idx = random.randint(0,self.img_num-1)
            im = cv2.imread(self.total_namelist[idx])
            if im is not None:
                top[0].data[i] = pre_process(im)
                for k in range(0, len(selected_attr)):
                    top[1].data[i][k] = self.attri_array[idx][int(selected_attr[k])]

    def backward(self, top, propagate_down, bottom):
        pass


class JointAttributeLayer(caffe.Layer):

    attri_num = 40
    point_num = 5

    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """
    train_total_namelist = []
    train_attri_array = []
    train_img_num = 0

    val_total_namelist = []
    val_attri_array =[]
    val_img_num = 0
    target_height = 224
    target_witdh = 224
    batch = 10
    train_imgset = 'align_train'
    val_imgset = 'align_val'

    def do_setup(self, imgset, attri_num):
        filename = os.path.join(datadir, imgset, imgset + 'list.txt')
        attrfile = os.path.join(datadir, imgset, imgset + '_attr.txt')
        det_df = read_csv(filename,sep=' ', header=None)
        total_namelist = det_df[det_df.columns[0]].values
        img_num = len(total_namelist)
        attri_array = np.zeros((img_num, attri_num))
        attr_f = open(attrfile, 'r')
        attr_line = attr_f.readline().strip().split()
        i = 0
        while attr_line:
            for j in range(0, attri_num):
                value = int(float(attr_line[j + 1]))
                assert value == 1 or value == -1
                attri_array[i, j] = value
            attr_line = attr_f.readline().strip().split()
            i += 1
        return total_namelist, img_num, attri_array


    def setup(self, bottom, top):
        self.train_total_namelist, self.train_img_num, self.train_attri_array = self.do_setup(self.train_imgset, self.attri_num)
        self.val_total_namelist, self.val_img_num, self.val_attri_array = self.do_setup(self.val_imgset, self.attri_num)
        top[0].reshape(2 * self.batch,3,self.target_height,self.target_witdh)
        top[1].reshape(2 * self.batch, len(selected_attr))

    def reshape(self, bottom, top):
        pass

    def do_forward(self, top, begin_index, total_namelist, img_num, attri_array):
        for i in range(self.batch):
            # print i
            idx = random.randint(0, img_num-1)
            im = cv2.imread(total_namelist[idx])
            if im is not None:
                top[0].data[i + begin_index] = pre_process(im)
                for k in range(0, len(selected_attr)):
                    top[1].data[i + begin_index][k] = attri_array[idx][int(selected_attr[k])]

    def forward(self, bottom, top):
         self.do_forward(top, 0, self.train_total_namelist, self.train_img_num,
                         self.train_attri_array)
         self.do_forward(top, self.batch, self.val_total_namelist, self.val_img_num,
                         self.val_attri_array)

    def backward(self, top, propagate_down, bottom):
        pass

