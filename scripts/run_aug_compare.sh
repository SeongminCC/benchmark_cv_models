# VGGNet16 - augmentation
python main.py \
--exp-name VGG16_aug \
--model VGG16 \
--opt-name SGD \
--aug-name weak \
--epochs 50 \
--batch-size 128 \
--use-wandb

# ResNet34 - augmentation
python main.py \
--exp-name ResNet34_aug \
--model ResNet34 \
--opt-name SGD \
--aug-name weak \
--epochs 50 \
--batch-size 128 \
--use-wandb

# ResNet50 - augmentation
python main.py \
--exp-name ResNet50_aug \
--model ResNet50 \
--opt-name SGD \
--aug-name weak \
--epochs 50 \
--batch-size 128 \
--use-wandb