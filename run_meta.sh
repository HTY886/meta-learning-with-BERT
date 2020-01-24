export XQUAD=$1

python run_meta_QA.py \
  --model_type bert \
  --do_train \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file $XQUAD \
  --predict_file $XQUAD/xquad.en.json \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 320 \
  --doc_stride 128 \
  --output_dir meta_model \
  --per_gpu_eval_batch_size=5   \
  --per_gpu_train_batch_size=5  \

