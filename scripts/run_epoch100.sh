# VGGNet16 - augmentation
python main.py \
--exp-name VGG16_epoch100 \
--model VGG16 \
--opt-name SGD \
--aug-name weak \
--epochs 100 \
--batch-size 128 \
--use-wandb

# ResNet34 - augmentation
python main.py \
--exp-name ResNet34_epoch100 \
--model ResNet34 \
--opt-name SGD \
--aug-name weak \
--epochs 100 \
--batch-size 128 \
--use-wandb

# ResNet50 - augmentation
python main.py \
--exp-name ResNet50_epoch100 \
--model ResNet50 \
--opt-name SGD \
--aug-name weak \
--epochs 100 \
--batch-size 128 \
--use-wandb