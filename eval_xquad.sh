XQUAD=$1
lang=$3

python run_squad.py \
  --model_type bert \
  --do_eval \
  --model_name_or_path $2 \
  --train_file $XQUAD/xquad.en.json \
  --predict_file $XQUAD"/xquad."$lang".json" \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 320 \
  --doc_stride 128 \
  --output_dir $2 \
  --per_gpu_eval_batch_size=5   \
  --per_gpu_train_batch_size=5  \


