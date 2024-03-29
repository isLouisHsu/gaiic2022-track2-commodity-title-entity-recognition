python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0-cosine \
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
    --use_last_n_layers=8 --agg_last_n_layers=mean \
    --do_rdrop --rdrop_weight=0.4 \
    --scheduler_type=cosine \
    --seed=42 \
    --fp16


# --do_lstm --num_lstm_layers=1 --lstm_dropout=0.5 \
# --adv_start_steps
# --swa_enable --swa_start=2 --swa_lr=1e-5 --swa_freq=1 --swa_anneal_epochs=100 --swa_anneal_strategy="linear" --logging_steps=500 --save_steps=500