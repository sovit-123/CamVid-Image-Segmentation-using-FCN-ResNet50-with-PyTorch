# CamVid Image Segmentation using FCN ResNet50 with PyTorch



## <u>About</u>

A PyTorch implementation of the CamVid dataset semantic segmentation using FCN ResNet50 FPN model. The dataset has been taken from **[CamVid (Cambridge-Driving Labeled Video Database)](https://www.kaggle.com/carlolepelaars/camvid)**.

![](https://github.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/blob/master/preview_images/preview1.jpg?raw=true)



## <u>Prerequires / Dependencies</u>

* PyTorch == 1.6.0
* Albumentations == 0.4.6
* Tensorboard == 1.1.5.0
* TensorboardX == 2.1



## <u>Running the Python Scripts</u>

* First make sure that you have given the correct `ROOT_PATH` in your `config.py`.
* Check every other parameters in `config.py` as per your requirement.

### To Train from Beginning (Training the First Time)

* To train the image segmentation model for the first time on the CamVid dataset:

  * ```
    python train.py --resume-training no
    ```

### To Continue Training 

* This is for those cases, if you stop training in between and want to resume again. **Make sure that while resuming training you provide more epochs to train for than the previous case.**

  * ```
    python train.py --resume-training yes --model <path to model weights>
    ```



## <u>Tensorboard Validation Log Results After Training for 75 Epochs</u>

* **Validation loss**

  ![](https://raw.githubusercontent.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/65e4ef87a3d2a39f568ab68a80f93cf5d946e8eb/preview_images/Valid_Loss.svg)

* **Validation Pixel Accuracy**

  ![](https://raw.githubusercontent.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/65e4ef87a3d2a39f568ab68a80f93cf5d946e8eb/preview_images/Valid_Pixel_Acc.svg)

* **Validation Mean IoU**

  ![](https://raw.githubusercontent.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/65e4ef87a3d2a39f568ab68a80f93cf5d946e8eb/preview_images/Valid_mIoU.svg)