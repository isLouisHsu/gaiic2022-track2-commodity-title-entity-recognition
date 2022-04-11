python prepare_data.py \
    --version=v4-ssl \
    --labeled_files \
        data/raw/train_data/train.txt \
    --unlabeled_files \
        data/raw/preliminary_test_a/word_per_line_preliminary_A.txt \
    --test_files \
        data/raw/preliminary_test_a/word_per_line_preliminary_A.txt \
    --output_dir=data/processed/ \
    --n_splits=1 \
    --seed=42

python run_span_classification_ssl.py \
    --experiment_code=nezha-50k-ssl-spanv1-datav2-lr3e-5-wd0.01-dropout0.1-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=outputs/gaiic_nezha_nezha-50k-spanv1-datav2-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3/checkpoint-eval_f1_micro_all_entity-best/ \
    --data_dir=data/processed/v4-ssl/ \
    --train_input_file=train.all.jsonl \
    --semi_input_file=semi.all.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --max_train_examples=1000 \
    --max_semi_examples=1000 \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_predict \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_test_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --semi_confident_thresh=0.9 \
    --semi_negative_rate=10.0 \
    --semi_loss_weight=1.0 \
    --semi_teacher_momentum=0.999 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.1 \
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
    --device_id=cpu
