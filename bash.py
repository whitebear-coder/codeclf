'''
python run.py --train_data_file=/home/linzexu/codeNet_clf --output_dir=./saved_models --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_train --num_train_epochs 50 \
    --block_size 256 --train_batch_size 8 --eval_batch_size 16 --learning_rate 2e-5 --max_grad_norm 1.0 --num_labels 10 --seed 123456  2>&1 | tee train.log
'''