import caffe
import os
import numpy as np
niter = 200
# losses will also be stored in the log
train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)

caffe.set_device(1)
caffe.set_mode_gpu()
# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('model/resnet_50/ResNet-50-model.caffemodel')
# solver.net.copy_from('../model/40_attri_vgg16_dlib_euclidean_iter_100.caffemodel')
solver.solve()
# We run the solver for niter times, and record the training loss.
# for it in range(niter):
#     solver.step(0)  # SGD by Caffe
#     # store the train loss
#     train_loss[it] = solver.net.blobs['loss'].data
#     if it % 10 == 0:
#         print 'iter %d, finetune_loss=%f' % (it, train_loss[it])
# print 'done'
