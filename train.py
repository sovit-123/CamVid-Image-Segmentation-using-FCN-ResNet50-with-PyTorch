import config
import argparse
import sys

from engine import Trainer
from dataset import train_dataset, train_data_loader
from dataset import valid_dataset, valid_data_loader
from model import model

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume-training', dest='resume_training',
                    required=True, help='whether to resume training or not',
                    choices=['yes', 'no'])
parser.add_argument('-p', '--model-path', dest='model_path',
                    help='path to trained model for resuming training')
args = vars(parser.parse_args())

epochs = config.EPOCHS

model.to(config.DEVICE)

if args['resume_training'] == 'yes':
    if args['model_path'] == None:
        sys.exit('\nPLEASE PROVIDE A MODEL TO RESUME TRAINING FROM!')
    trainer = Trainer( 
    model, 
    train_data_loader, 
    train_dataset,
    valid_data_loader,
    valid_dataset,
    config.CLASSES_TO_TRAIN,
    epochs,
    args['resume_training'],
    model_path=args['model_path']
)

else:
    trainer = Trainer( 
        model, 
        train_data_loader, 
        train_dataset,
        valid_data_loader,
        valid_dataset,
        config.CLASSES_TO_TRAIN,
        epochs,
        args['resume_training']
    )

trained_epochs = trainer.get_num_epochs()
epochs_to_train = epochs - trained_epochs

train_loss , train_mIoU = [], []
valid_loss , valid_mIoU = [], []
for epoch in range(epochs_to_train):
    print(f"Epoch {epoch+1+trained_epochs} of {epochs}")
    train_epoch_loss, train_epoch_mIoU = trainer.fit()
    valid_epoch_loss, valid_epoch_mIoU = trainer.validate(epoch+1+trained_epochs)
    train_loss.append(train_epoch_loss)
    train_mIoU.append(train_epoch_mIoU)
    valid_loss.append(valid_epoch_loss)
    valid_mIoU.append(valid_epoch_mIoU)
    print(f"Train Loss: {train_epoch_loss:.4f}, Train mIoU: {train_epoch_mIoU:.4f}")
    print(f'Valid Loss: {valid_epoch_loss:.4f}, Valid mIoU: {valid_epoch_mIoU:.4f}')

    # save model every 5 epochs
    if (epoch+1+trained_epochs) % 5 == 0:
        print('SAVING MODEL')
        trainer.save_model(epoch+1+trained_epochs)
        print('SAVING COMPLETE')
        print('TRAINING COMPLETE')