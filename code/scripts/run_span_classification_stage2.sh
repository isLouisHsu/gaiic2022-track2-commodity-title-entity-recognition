# 10折划分，A榜最优单模型（无伪标签），复赛基线
python prepare_data.py \
    --version=stage2-v0 \
    --labeled_files \
        ../data/contest_data/train_data/train.txt \
    --test_files \
        ../data/contest_data/preliminary_test_b/word_per_line_preliminary_B.txt \
    --output_dir=../data/tmp_data/ \
    --n_splits=10 \
    --seed=42
python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --do_rdrop \
    --rdrop_weight=0.3 \
    --seed=42 \
    --fp16
# 2022-04-27 21:51:39 - INFO - root -   eval_f1_micro_all_entity = 0.8131
# 2022-04-27 21:51:39 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8657
# 2022-04-27 21:51:39 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8413
# 2022-04-27 21:51:39 - INFO - root -   eval_f1_micro_without_label_entity = 0.905
# 2022-04-27 21:51:39 - INFO - root -   eval_loss = 0.0105

# MacNeZha
python run_span_classification_v1.py \
    --experiment_code=macnezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-mac-seq128-lr2e-5-mlm0.15-4gram-100k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --do_rdrop \
    --rdrop_weight=0.3 \
    --seed=42 \
    --fp16
# 2022-04-27 21:48:03 - INFO - root -   eval_f1_micro_all_entity = 0.8121
# 2022-04-27 21:48:03 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8647
# 2022-04-27 21:48:03 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8407
# 2022-04-27 21:48:03 - INFO - root -   eval_f1_micro_without_label_entity = 0.9044
# 2022-04-27 21:48:03 - INFO - root -   eval_loss = 0.0105

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr5e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --do_rdrop \
    --rdrop_weight=0.3 \
    --seed=42 \
    --fp16
# 2022-04-27 21:49:50 - INFO - root -   eval_f1_micro_all_entity = 0.8121
# 2022-04-27 21:49:50 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8655
# 2022-04-27 21:49:50 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8404
# 2022-04-27 21:49:50 - INFO - root -   eval_f1_micro_without_label_entity = 0.9051
# 2022-04-27 21:49:50 - INFO - root -   eval_loss = 0.0108

# 去掉R-Drop，用于对比实验，54类
python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --seed=42 \
    --fp16
# 2022-04-28 23:58:55 - INFO - root -   eval_f1_micro_all_entity = 0.8127
# 2022-04-28 23:58:55 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8651
# 2022-04-28 23:58:55 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8411
# 2022-04-28 23:58:55 - INFO - root -   eval_f1_micro_without_label_entity = 0.9047
# 2022-04-28 23:58:55 - INFO - root -   eval_loss = 0.0105

# wwm-4gram-200k
export EXCLUDE_NOT_EXIST_LABELS=false
python run_span_classification_v1.py \
    --experiment_code=nezha-4gram-200k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-4gram-seq128-lr2e-5-mlm0.15-200k-warmup5k-bs64x2/checkpoint-200000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --seed=42 \
    --fp16
# 2022-04-30 12:55:08 - INFO - root -   eval_f1_micro_all_entity = 0.812
# 2022-04-30 12:55:08 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8645
# 2022-04-30 12:55:08 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8403
# 2022-04-30 12:55:08 - INFO - root -   eval_f1_micro_without_label_entity = 0.9039
# 2022-04-30 12:55:08 - INFO - root -   eval_loss = 0.0105

# nezhaxy, wwm-4gram-200k
nohup \
python run_span_classification_v1.py \
    --experiment_code=nezha-4gram-200k-spanv1xy-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezhaxy \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-4gram-seq128-lr2e-5-mlm0.15-200k-warmup5k-bs64x2/checkpoint-200000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --seed=42 \
    --fp16 \
> ../data/tmp_data/nohup.out &
# share_fc
# 2022-04-30 11:56:59 - INFO - root -   eval_f1_micro_all_entity = 0.811
# 2022-04-30 11:56:59 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8655
# 2022-04-30 11:56:59 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8404
# 2022-04-30 11:56:59 - INFO - root -   eval_f1_micro_without_label_entity = 0.9048
# 2022-04-30 11:56:59 - INFO - root -   eval_loss = 0.0169

export EXCLUDE_NOT_EXIST_LABELS=true
python run_span_classification_v1.py \
    --experiment_code=nezha-4gram-200k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs8x2-sinusoidal-biaffine-fgm1.0-rdrop0.3 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-4gram-seq128-lr2e-5-mlm0.15-200k-warmup5k-bs64x2/checkpoint-200000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_test_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.3 \
    --seed=42 \
    --fp16
# 2022-05-01 16:51:30 - INFO - root -   eval_f1_micro_all_entity = 0.8108
# 2022-05-01 16:51:30 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8641
# 2022-05-01 16:51:30 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8391
# 2022-05-01 16:51:30 - INFO - root -   eval_f1_micro_without_label_entity = 0.9039
# 2022-05-01 16:51:30 - INFO - root -   eval_loss = 0.0109

export EXCLUDE_NOT_EXIST_LABELS=true
python run_span_classification_v1.py \
    --experiment_code=nezha-4gram-200k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.4-span35-e6-bs8x4-sinusoidal-biaffine-fgm1.0-rdrop1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-4gram-seq128-lr2e-5-mlm0.15-200k-warmup5k-bs64x2/checkpoint-200000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_test_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.4 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=1.0 \
    --seed=42 \
    --fp16
# 2022-05-02 14:07:07 - INFO - root -   eval_f1_micro_all_entity = 0.8126
# 2022-05-02 14:07:07 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8651
# 2022-05-02 14:07:07 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8407
# 2022-05-02 14:07:07 - INFO - root -   eval_f1_micro_without_label_entity = 0.9043
# 2022-05-02 14:07:07 - INFO - root -   eval_loss = 0.0105

nohup \
python run_span_classification_v1.py \
    --experiment_code=nezha-4gram-200k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.4-span35-e6-bs8x4-sinusoidal-biaffine-fgm1.0-rdrop1.0-adam1e-6 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-4gram-seq128-lr2e-5-mlm0.15-200k-warmup5k-bs64x2/checkpoint-200000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_test_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.4 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=1.0 \
    --seed=42 \
    --fp16 \
> ../data/tmp_data/nohup.out &
# 2022-05-02 16:10:42 - INFO - root -   eval_f1_micro_all_entity = 0.8119
# 2022-05-02 16:10:42 - INFO - root -   eval_f1_micro_all_entity_fx = 0.864
# 2022-05-02 16:10:42 - INFO - root -   eval_f1_micro_all_entity_fy = 0.84
# 2022-05-02 16:10:42 - INFO - root -   eval_f1_micro_without_label_entity = 0.9034
# 2022-05-02 16:10:42 - INFO - root -   eval_loss = 0.0104