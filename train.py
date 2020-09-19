import config

from engine import Trainer
from dataset import train_dataset, train_data_loader
from dataset import valid_dataset, valid_data_loader
from model import model

epochs = config.EPOCHS

model.to(config.DEVICE)

trainer = Trainer( 
    model, 
    train_data_loader, 
    train_dataset,
    valid_data_loader,
    valid_dataset,
    config.CLASSES_TO_TRAIN
)

train_loss , train_mIoU = [], []
valid_loss , valid_mIoU = [], []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_mIoU = trainer.fit(epoch, epochs)
    valid_epoch_loss, valid_epoch_mIoU = trainer.validate(epoch, epochs)
    train_loss.append(train_epoch_loss)
    train_mIoU.append(train_epoch_mIoU)
    valid_loss.append(valid_epoch_loss)
    valid_mIoU.append(valid_epoch_mIoU)
    print(f"Train Loss: {train_epoch_loss:.4f}, Train mIoU: {train_epoch_mIoU:.4f}")
    print(f'Valid Loss: {valid_epoch_loss:.4f}, Valid mIoU: {valid_epoch_mIoU:.4f}')

    # save model every 5 epochs
    if (epoch+1) % 5 == 0:
        print('SAVING MODEL')
        trainer.save_model(epoch)
        print('SAVING COMPLETE')
        print('TRAINING COMPLETE')