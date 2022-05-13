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

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --label_smoothing=0.01 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-06 23:22:21 - INFO - root -   eval_f1_micro_all_entity = 0.8136
# 2022-05-06 23:22:21 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8656
# 2022-05-06 23:22:21 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8413
# 2022-05-06 23:22:21 - INFO - root -   eval_f1_micro_without_label_entity = 0.9044
# 2022-05-06 23:22:21 - INFO - root -   eval_loss = 0.1042

python prepare_data.py \
    --version=stage2-v1 \
    --labeled_files \
        ../data/contest_data/train_data/train.txt \
    --test_files \
        ../data/contest_data/preliminary_test_b/word_per_line_preliminary_B.txt \
    --unlabeled_files \
        ../data/contest_data/train_data/unlabeled_train_data.txt \
    --start_unlabeled_files=0 \
    --end_unlabeled_files=40000 \
    --output_dir=../data/tmp_data/ \
    --n_splits=10 \
    --seed=42
python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v1.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01-pseuv0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v1/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --label_smoothing=0.01 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --pseudo_input_file=semi.0:40000.jsonl \
    --pseudo_teachers_name_or_path \
        ../data/model_data/gaiic_nezha_nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3/checkpoint-eval_f1_micro_all_entity-best \
        ../data/model_data/gaiic_nezha_nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01/checkpoint-eval_f1_micro_all_entity-best \
    --pseudo_temperature=0.5 \
    --pseudo_weight=0.5 \
    --seed=42 \
    --fp16
# 2022-05-07 12:00:31 - INFO - root -   eval_f1_micro_all_entity = 0.8136
# 2022-05-07 12:00:31 - INFO - root -   eval_f1_micro_all_entity_fx = 0.866
# 2022-05-07 12:00:31 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8411
# 2022-05-07 12:00:31 - INFO - root -   eval_f1_micro_without_label_entity = 0.9048
# 2022-05-07 12:00:31 - INFO - root -   eval_loss = 0.1042

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01-swav0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --checkpoint_predict_code=checkpoint-swa_eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.01 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --swa_enable --swa_start=0.9 --swa_lr=1e-6 --swa_freq=1 \
    --swa_anneal_epochs=10 --swa_anneal_strategy=linear \
    --fp16
# 2022-05-07 13:35:39 - INFO - root -   eval_f1_micro_all_entity = 0.8132
# 2022-05-07 13:35:39 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8652
# 2022-05-07 13:35:39 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8412
# 2022-05-07 13:35:39 - INFO - root -   eval_f1_micro_without_label_entity = 0.9045
# 2022-05-07 13:35:39 - INFO - root -   eval_loss = 0.1042
# 2022-05-07 13:34:53 - INFO - root -   swa_eval_f1_micro_all_entity = 0.8135
# 2022-05-07 13:34:53 - INFO - root -   swa_eval_f1_micro_all_entity_fx = 0.8654
# 2022-05-07 13:34:53 - INFO - root -   swa_eval_f1_micro_all_entity_fy = 0.8413
# 2022-05-07 13:34:53 - INFO - root -   swa_eval_f1_micro_without_label_entity = 0.9045

python run_span_classification_v1.py \
    --experiment_code=nezha-finetune-spanv1-datas2v0.0-lr1e-5-wd0.01-dropout0.3-span35-e1-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01-swav0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/model_data/gaiic_nezha_nezha-100k-spanv1-datas2v1.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01-pseuv0/checkpoint-eval_f1_micro_all_entity-best/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-5 \
    --weight_decay=0.01 \
    --num_train_epochs=1 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.3 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.01 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --swa_enable --swa_start=0.9 --swa_lr=1e-6 --swa_freq=1 \
    --swa_anneal_epochs=10 --swa_anneal_strategy=linear \
    --seed=42 \
    --fp16
# 2022-05-07 22:23:56 - INFO - root -   eval_f1_micro_all_entity = 0.813
# 2022-05-07 22:23:56 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8651
# 2022-05-07 22:23:56 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8409
# 2022-05-07 22:23:56 - INFO - root -   eval_f1_micro_without_label_entity = 0.9042
# 2022-05-07 22:23:56 - INFO - root -   eval_loss = 0.1041
# 2022-05-07 22:23:56 - INFO - root -   swa_eval_f1_micro_all_entity = 0.8129
# 2022-05-07 22:23:56 - INFO - root -   swa_eval_f1_micro_all_entity_fx = 0.865
# 2022-05-07 22:23:56 - INFO - root -   swa_eval_f1_micro_all_entity_fy = 0.8408
# 2022-05-07 22:23:56 - INFO - root -   swa_eval_f1_micro_without_label_entity = 0.9041

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr2e-5-wd0.001-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4 \
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
    --learning_rate=2e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.001 \
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
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-08 00:48:20 - INFO - root -   eval_f1_micro_all_entity = 0.812
# 2022-05-08 00:48:20 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8641
# 2022-05-08 00:48:20 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8403
# 2022-05-08 00:48:20 - INFO - root -   eval_f1_micro_without_label_entity = 0.9035
# 2022-05-08 00:48:20 - INFO - root -   eval_loss = 0.0104

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01 \
    --task_name=gaiic \
    --model_type=nezhagp \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --use_rope --pe_dim=64 --pe_max_len=512 \
    --label_smoothing=0.01 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --adv_enable --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-08 13:49:28 - INFO - root -   eval_f1_micro_all_entity = 0.8134
# 2022-05-08 13:49:28 - INFO - root -   eval_f1_micro_all_entity_fx = 0.865
# 2022-05-08 13:49:28 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8411
# 2022-05-08 13:49:28 - INFO - root -   eval_f1_micro_without_label_entity = 0.9037
# 2022-05-08 13:49:28 - INFO - root -   eval_loss = 0.1054

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.001-last4mean-lstm2-dropout0.3-span128-e8-bs16x1-fgm1.0-rdrop0.4 \
    --task_name=gaiic \
    --model_type=nezhagp \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.001 \
    --num_train_epochs=8 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --max_span_length=128 \
    --classifier_dropout=0.3 \
    --use_last_n_layers=4 --agg_last_n_layers=mean \
    --do_lstm --num_lstm_layers=2 --lstm_dropout=0.5 \
    --use_rope --pe_dim=64 --pe_max_len=512 \
    --loss_type=ce \
    --decode_thresh=0.0 \
    --adv_enable --adv_type=fgm --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-08 16:13:16 - INFO - root -   eval_f1_micro_all_entity = 0.8127
# 2022-05-08 16:13:16 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8643
# 2022-05-08 16:13:16 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8405
# 2022-05-08 16:13:16 - INFO - root -   eval_f1_micro_without_label_entity = 0.9035
# 2022-05-08 16:13:16 - INFO - root -   eval_loss = 0.0087

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-lsr0.01 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --label_smoothing=0.01 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_name=weight --adv_type=awp --adv_alpha=1.0 --adv_epsilon=0.001 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-09 23:13:04 - INFO - root -   eval_f1_micro_all_entity = 0.8144
# 2022-05-09 23:13:04 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8672
# 2022-05-09 23:13:04 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8415
# 2022-05-09 23:13:04 - INFO - root -   eval_f1_micro_without_label_entity = 0.9055
# 2022-05-09 23:13:04 - INFO - root -   eval_loss = 0.1041

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce0.1 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --loss_type=ploy1_ce --ploy1_epsilon=1.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_name=weight --adv_type=awp --adv_alpha=1.0 --adv_epsilon=0.001 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-11 23:29:29 - INFO - root -   eval_f1_micro_all_entity = 0.8149
# 2022-05-11 23:29:29 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8675
# 2022-05-11 23:29:29 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8424
# 2022-05-11 23:29:29 - INFO - root -   eval_f1_micro_without_label_entity = 0.9061
# 2022-05-11 23:29:29 - INFO - root -   eval_loss = 0.015
# 2022-05-11 23:29:29 - INFO - packages -  Steps 6750: Metric eval_f1_micro_all_entity improved from 0.8145 to 0.8149. New best score: 0.8149

python data_augmentation.py \
    --input_file=../data/tmp_data/stage2-v0/train.0.jsonl \
    --output_file=../data/tmp_data/stage2-v0/train.0.augv0.jsonl \
    --augment_times=1 \
    --do_exchange_entity_augment \
    --seed=42
    
python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --loss_type=ploy1_ce --ploy1_epsilon=2.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_name=weight --adv_type=awp --adv_alpha=1.0 --adv_epsilon=0.001 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-13 00:06:56 - INFO - root -   eval_f1_micro_all_entity = 0.8152
# 2022-05-13 00:06:56 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8677
# 2022-05-13 00:06:56 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8428
# 2022-05-13 00:06:56 - INFO - root -   eval_f1_micro_without_label_entity = 0.9063
# 2022-05-13 00:06:56 - INFO - root -   eval_loss = 0.0193

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-augv0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.augv0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --loss_type=ploy1_ce --ploy1_epsilon=2.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_name=weight --adv_type=awp --adv_alpha=1.0 --adv_epsilon=0.001 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-13 11:55:02 - INFO - root -   eval_f1_micro_all_entity = 0.8127
# 2022-05-13 11:55:02 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8635
# 2022-05-13 11:55:02 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8408
# 2022-05-13 11:55:02 - INFO - root -   eval_f1_micro_without_label_entity = 0.9034
# 2022-05-13 11:55:02 - INFO - root -   eval_loss = 0.0198

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgme1.0-rdrop0.4-ploy1_ce2.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --loss_type=ploy1_ce --ploy1_epsilon=2.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_name=embeddings --adv_type=fgm --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-13 13:40:11 - INFO - root -   eval_f1_micro_all_entity = 0.8118
# 2022-05-13 13:40:11 - INFO - root -   eval_f1_micro_all_entity_fx = 0.864
# 2022-05-13 13:40:11 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8397
# 2022-05-13 13:40:11 - INFO - root -   eval_f1_micro_without_label_entity = 0.9029
# 2022-05-13 13:40:11 - INFO - root -   eval_loss = 0.0199

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-pgde1.0-rdrop0.4-ploy1_ce2.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --loss_type=ploy1_ce --ploy1_epsilon=2.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_name=embeddings --adv_type=pgd --adv_alpha=0.3 --adv_epsilon=1.0 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16
# 2022-05-13 15:26:14 - INFO - root -   eval_f1_micro_all_entity = 0.8136
# 2022-05-13 15:26:14 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8656
# 2022-05-13 15:26:14 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8413
# 2022-05-13 15:26:14 - INFO - root -   eval_f1_micro_without_label_entity = 0.9045
# 2022-05-13 15:26:14 - INFO - root -   eval_loss = 0.0198

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --classifier_dropout=0.5 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --loss_type=ploy1_ce --ploy1_epsilon=2.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_name=weight --adv_type=awp --adv_alpha=1.0 --adv_epsilon=0.001 \
    --do_rdrop --rdrop_weight=0.4 \
    --seed=42 \
    --fp16

python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop2.0-ploy1_ce2.5 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/stage2-v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_eval \
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
    --loss_type=ploy1_ce --ploy1_epsilon=2.5 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable --adv_name=weight --adv_type=awp --adv_alpha=1.0 --adv_epsilon=0.001 \
    --do_rdrop --rdrop_weight=2.0 \
    --seed=42 \
    --fp16
# 2022-05-14 00:17:14 - INFO - root -   eval_f1_micro_all_entity = 0.8151
# 2022-05-14 00:17:14 - INFO - root -   eval_f1_micro_all_entity_fx = 0.8681
# 2022-05-14 00:17:14 - INFO - root -   eval_f1_micro_all_entity_fy = 0.8428
# 2022-05-14 00:17:14 - INFO - root -   eval_f1_micro_without_label_entity = 0.9068
# 2022-05-14 00:17:14 - INFO - root -   eval_loss = 0.0214

# TODO: 更新线上代码