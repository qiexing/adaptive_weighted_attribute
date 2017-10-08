import caffe
import numpy as np


class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num



class TrainValWeightedEuclideanLossLayer(caffe.Layer):
    """
    Compute the Wegithed Loss
    """
    count = 0
    batch = 0
    interval = 200
    val_loss = []
    pre_val_mean = 0
    norm_trend = None
    norm_loss = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.batch = bottom[0].num / 2  # Indicate first half as train, second half as val
        self.count = 0
        self.norm_trend = np.ones(bottom[0].data.shape[1])
        self.norm_loss = np.ones(bottom[0].data.shape[1])

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        self.val_loss.append(np.sum(self.diff[self.batch:]**2, axis=0))
        top[0].data[...] = np.sum(self.diff[0:self.batch]**2) / self.batch / 2.
        self.count += 1

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            if self.count == self.interval:
                self.pre_val_mean = np.mean(self.val_loss[0:self.interval], axis=0)
            if self.count >= 2 * self.interval and self.count % self.interval == 0:
                begin_index = self.count - self.interval
                end_index = self.count
                cur_val_mean = np.mean(self.val_loss[begin_index:end_index], axis=0)
                trend = abs(cur_val_mean - self.pre_val_mean) / cur_val_mean
                self.norm_trend = trend / np.mean(trend)
                self.norm_loss = cur_val_mean / np.mean(cur_val_mean)
            weights = self.norm_trend * self.norm_loss
            norm_weights = weights / np.mean(weights)
            repmated_weight = np.tile(norm_weights, [self.batch, 1])
            bottom[i].diff[0:self.batch] = sign * repmated_weight * self.diff[0:self.batch] / self.batch
