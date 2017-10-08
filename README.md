## adaptively\_weighted\_attribute
This is the code for our paper: adaptively weighted multi-task deep network for person attribute classification. If you find the project useful in your research, please consider citing.


### Requirements 

- `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config
  WITH_PYTHON_LAYER := 1
  ```

### Prepare

- You can download the ResNet50 model and CelebA train, val, test files from [google drive](https://drive.google.com/open?id=0B3S9YfMB_24QZENmaXYzdTFvdkE) or [baidu yun](http://pan.baidu.com/s/1hrJUedm)
-  put the **resnet_50** folder under **model** folder
-  put the **CelebA** folder under **data** folder.

### Train

- Command : python train_model.py



