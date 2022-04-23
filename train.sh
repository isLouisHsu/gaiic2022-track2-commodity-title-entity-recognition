cd code/

# 准备预训练语料
version=pretrain-v1
python prepare_corpus.py \
    --version=${version} \
    --output_dir=../data/tmp_data/ \
    --min_length=0 \
    --max_length=128 \
    --train_ratio=0.9 \
    --seed=42
for data_type in train valid
do
python run_chinese_ref.py \
    --file_name=../data/tmp_data/${version}/corpus.${data_type}.txt \
    --bert=../data/pretrain_model/nezha-cn-base/vocab.txt \
    --seg_save_path=../data/tmp_data/${version}/seg.${data_type}.txt \
    --ref_save_path=../data/tmp_data/${version}/ref.${data_type}.txt
done
# 预训练
export WANDB_DISABLED=true
data_dir=../data/tmp_data/pretrain-v1
version=nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2
python run_mlm_wwm.py \
    --model_name_or_path=../data/pretrain_model/nezha-cn-base/ \
    --model_type=nezha \
    --train_file=${data_dir}/corpus.train.txt \
    --validation_file=${data_dir}/corpus.valid.txt \
    --train_ref_file=${data_dir}/ref.train.txt \
    --validation_ref_file=${data_dir}/ref.valid.txt \
    --cache_dir=cache/ \
    --overwrite_cache \
    --max_seq_length=128 \
    --preprocessing_num_workers=8 \
    --mlm_probability=0.15 \
    --output_dir=../data/pretrain_model/${version}/ \
    --do_train --do_eval \
    --warmup_steps=3000 \
    --max_steps=100000 \
    --evaluation_strategy=steps \
    --eval_steps=2000 \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --label_smoothing_factor=0.0 \
    --learning_rate=2e-5 \
    --weight_decay=0.01 \
    --logging_dir=outputs/${version}/log/ \
    --logging_strategy=steps \
    --logging_steps=2000 \
    --save_strategy=steps \
    --save_steps=2000 \
    --save_total_limit=20 \
    --dataloader_num_workers=4 \
    --seed=42

# 微调数据
python prepare_data.py \
    --version=v3 \
    --labeled_files \
        ../data/contest_data/train_data/train.txt \
    --test_files \
        ../data/contest_data/preliminary_test_a/word_per_line_preliminary_A.txt \
    --output_dir=../data/tmp_data/ \
    --n_splits=1 \
    --seed=42
# 线上0.8136793661222608
python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datav3-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/v3/ \
    --train_input_file=train.all.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_predict \
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
    --seed=42

# 伪标签
python prepare_data.py \
    --version=v5-ssl \
    --labeled_files \
        ../data/contest_data/train_data/train.txt \
    --unlabeled_files \
        ../data/contest_data/preliminary_test_a/sample_per_line_preliminary_A.txt \
        ../data/contest_data/train_data/unlabeled_train_data.txt \
    --test_files \
        ../data/contest_data/preliminary_test_a/word_per_line_preliminary_A.txt \
    --output_dir=../data/tmp_data/ \
    --n_splits=1 \
    --start_unlabeled_files=0 \
    --end_unlabeled_files=10000 \
    --seed=42
## 2. 推断，得到标注
python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datav3-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/v5-ssl/ \
    --train_input_file=train.all.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=semi.0:10000.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
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
    --seed=42

# 第二阶段微调
pseudo_dir=../data/model_data/gaiic_nezha_nezha-100k-spanv1-datav3-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3/checkpoint-eval_f1_micro_all_entity-best
python prepare_data.py \
    --version=v6-pl \
    --labeled_files \
        ../data/contest_data/train_data/train.txt \
    --pseudo_files \
        ${pseudo_dir}/semi.0:10000.jsonl.predictions.txt \
    --test_files \
        ../data/contest_data/preliminary_test_a/word_per_line_preliminary_A.txt \
    --output_dir=../data/tmp_data/ \
    --n_splits=1 \
    --seed=42
python run_span_classification_v1.py \
    --experiment_code=nezha-100k-spanv1-datav6-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3-pseu0.4 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/ \
    --data_dir=../data/tmp_data/v6-pl/ \
    --train_input_file=train.all.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=../data/model_data/ \
    --do_train --do_predict \
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
