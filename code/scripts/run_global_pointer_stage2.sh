# python exp_gaiic_global_pointer_v2.py \
#    --experiment_code=experiment_bert_base_fold0_gp_v2_pre_v62 \
#    --task_name=gaiic \
#    --model_type=nezha \
#    --do_lower_case \
#    --pretrained_model_path=../data/submission/checkpoint-7800 \
#    --data_dir=../data/tmp_data/10_folds_data/ \
#    --train_input_file=train.0.jsonl \
#    --eval_input_file=dev.0.jsonl \
#    --output_dir=../data/tmp_data/ \
#    --do_predict_test \
#    --test_input_file=../test_submit_dev_0.txt \
#    --eval_checkpoint_path=../data/best_model \
#    --submit_file_path=../results.txt \
#    --evaluate_during_training \
#    --train_max_seq_length=128 \
#    --eval_max_seq_length=128 \
#    --test_max_seq_length=128 \
#    --per_gpu_train_batch_size=16 \
#    --per_gpu_eval_batch_size=32 \
#    --per_gpu_test_batch_size=32 \
#    --gradient_accumulation_steps=1 \
#    --learning_rate=3e-5 \
#    --other_learning_rate=1e-3 \
#    --weight_decay=0.001 \
#    --scheduler_type=cosine \
#    --base_model_name=bert \
#    --warmup_proportion=0.1 \
#    --max_grad_norm=1.0 \
#    --num_train_epochs=10 \
#    --use_rope \
#    --do_lstm \
#    --do_fgm \
#    --num_lstm_layers=2 \
#    --adam_epsilon=1e-8 \
#    --post_lstm_dropout=0.5 \
#    --inner_dim=64 \
#    --loss_type=pcl \
#    --pcl_epsilon=2.5 \
#    --pcl_alpha=1.5 \
#    --do_awp \
#    --awp_f1=0.810 \
#    --awp_lr=0.1 \
#    --do_rdrop \
#    --rdrop_weight=0.4 \
#    --rdrop_epoch=1 \
#    --seed=42

python exp_gaiic_global_pointer_v2.py \
   --experiment_code=experiment_bert_base_fold0_gp_v2_pre_v62 \
   --task_name=gaiic \
   --model_type=nezha \
   --do_lower_case \
   --pretrained_model_path=../data/pretrain_model/nezha_cn_base_1.2/ \
   --data_dir=../data/tmp_data/stage2-gp/ \
   --train_input_file=train.0.jsonl \
   --eval_input_file=dev.0.jsonl \
   --output_dir=../data/model_data/ \
   --do_train \
   --test_input_file=../test_submit_dev_0.txt \
   --eval_checkpoint_path=../data/best_model \
   --submit_file_path=../results.txt \
   --evaluate_during_training \
   --train_max_seq_length=128 \
   --eval_max_seq_length=128 \
   --test_max_seq_length=128 \
   --per_gpu_train_batch_size=16 \
   --per_gpu_eval_batch_size=32 \
   --per_gpu_test_batch_size=32 \
   --gradient_accumulation_steps=1 \
   --learning_rate=3e-5 \
   --other_learning_rate=1e-3 \
   --weight_decay=0.001 \
   --scheduler_type=cosine \
   --base_model_name=bert \
   --warmup_proportion=0.1 \
   --max_grad_norm=1.0 \
   --num_train_epochs=10 \
   --use_rope \
   --do_lstm \
   --do_fgm \
   --num_lstm_layers=2 \
   --adam_epsilon=1e-8 \
   --post_lstm_dropout=0.5 \
   --inner_dim=64 \
   --loss_type=pcl \
   --pcl_epsilon=2.5 \
   --pcl_alpha=1.5 \
   --do_awp \
   --awp_f1=0.810 \
   --awp_lr=0.1 \
   --do_rdrop \
   --rdrop_weight=0.4 \
   --rdrop_epoch=1 \
   --seed=42
# 2022-05-18 22:02:07 - INFO - root -   f1 = 0.81638
# 2022-05-18 22:02:07 - INFO - root -   acc = 0.81594
# 2022-05-18 22:02:07 - INFO - root -   recall = 0.81692
# 2022-05-18 22:02:07 - INFO - root -   loss = 0.02501
