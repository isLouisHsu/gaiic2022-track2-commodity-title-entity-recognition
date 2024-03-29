test_file=../${1}
echo ${test_file}

cd code/
python prepare_data.py \
    --version=v6-pl-test \
    --labeled_files \
        ../data/contest_data/train_data/train.txt \
    --test_files \
        ${test_file} \
    --output_dir=../data/tmp_data/ \
    --n_splits=1 \
    --seed=42
python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datav6-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3-pseu0.4 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/v6-pl-test/ \
    --train_input_file=train.all.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_B.jsonl \
    --do_lower_case \
    --output_dir=../data/best_model/ \
    --do_predict \
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
    --pseudo_input_file=pseudo.jsonl \
    --pseudo_weight=0.4 \
    --seed=42 \
    --fp16
cp ../data/best_model/gaiic_nezha_nezha-100k-spanv1-datav6-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3-pseu0.4/checkpoint-eval_f1_micro_all_entity-best/word_per_line_preliminary_B.jsonl.predictions.txt ../data/submission/result.txt
