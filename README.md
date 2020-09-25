# CamVid Image Segmentation using FCN ResNet50 with PyTorch



## <u>Table of Contents</u>

* [About the Project](#About-the-Project)
* [TODO](#TODO)
* [Prerequisites / Dependencies](#Prerequisites-/-Dependencies)
* [Running the Python Scripts](#Running-the-Python-Scripts)
* [Tensorboard Validation Log Results After Training for 75 Epochs](#Tensorboard-Validation-Log-Results-After-Training-for-75-Epochs)
* [Some Useful Features of This Project](#Some-Useful-Features-of-This-Project)



## <u>About the Project</u>

A PyTorch implementation of the CamVid dataset semantic segmentation using FCN ResNet50 FPN model. The dataset has been taken from **[CamVid (Cambridge-Driving Labeled Video Database)](https://www.kaggle.com/carlolepelaars/camvid)**.

![](https://github.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/blob/master/preview_images/preview1.jpg?raw=true)



## <u>TODO</u>

- [x] Add functionality for easier resuming of training.

- [x] Add Tensorboard logging.

- [x] Add functionality for continuation of Tensorboard graph logging when resuming training.

- [x] Add batch-wise metrics updation in terminal. Like the following.

  ```
  Epoch 1 of 75
  Training
  Loss: 1.6733 | mIoU: 0.1080 | PixAcc: 0.7444: : 24it [02:32,  6.35s/it]
  Validating
  Loss: 1.3548 | mIoU: 0.0900 | PixAcc: 0.7415: : 7it [00:24,  3.48s/it]
  ```

- [ ] Add PyTorch AMP support.



## <u>Prerequisites / Dependencies</u>

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



## <u>Some Useful Features of This Project</u>

* ***This project enables batch-wise updation of metrics with `tqdm` progress bar in the terminal instead of epoch-wise metrics updation. This helps in easier tracking and debugging of the metrics and we do not need to wait for a whole epoch to know whether our model is learning or not.***

* ***When we resume training, then although a new Tensorboard run is created but the graph will continue from the previous training epochs.*** 

* ***If you want, you can use this code repo as a semantic segmentation framework code base for any new dataset. You just need to bring the new dataset to the dataset format as per CamVid. The following is the CamVid data directory format.***

  ```
  ├───test # contains all the test images
  ├───test_labels # contains all the test segmentation images
  ├───train # contains all the train images
  ├───train_labels # contains all the train segmentation images
  ├───val # contains all the validation images
  └───val_labels # contains all the validation segmentation images
  ```

  