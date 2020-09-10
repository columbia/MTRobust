optim="adam"
GPUID=0
lambda=0.01
# single task
CUDE_VISIBLE_DEVICES=$GPUID python mtasks_train.py --dataset taskonomy --model resnet18 --customize_class --class_to_train E --class_to_test E --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim $optim --mt_lambda $lambda
# multitask
CUDE_VISIBLE_DEVICES=$GPUID python mtasks_train.py --dataset taskonomy --model resnet18 --customize_class --class_to_train Es --class_to_test E --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim $optim --mt_lambda $lambda

