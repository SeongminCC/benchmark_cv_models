# my_model
python main.py \
--exp-name my_model_SGD \
--model my_model \
--opt-name SGD \
--aug-name default \
--epochs 50 \
--batch-size 128 \
--use-wandb

# ResNet18 (SGD)
python main.py \
--exp-name ResNet18_SGD \
--model ResNet18 \
--opt-name SGD \
--aug-name default \
--epochs 50 \
--batch-size 128 \
--use-wandb

# ResNet34 (SGD)
python main.py \
--exp-name ResNet34_SGD \
--model ResNet34 \
--opt-name SGD \
--aug-name default \
--epochs 50 \
--batch-size 128 \
--use-wandb

# ResNet50 (SGD)
python main.py \
--exp-name ResNet50_SGD \
--model ResNet50 \
--opt-name SGD \
--aug-name default \
--epochs 50 \
--batch-size 128 \
--use-wandb
