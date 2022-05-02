python prepare_corpus.py \
    --version=pretrain-v0 \
    --output_dir=data/processed/ \
    --min_length=0 \
    --max_length=256 \
    --train_ratio=0.8 \
    --seed=42

for data_type in train valid
do
python run_chinese_ref.py \
    --file_name=data/processed/pretrain-v0/corpus.${data_type}.txt \
    --ltp=/home/louishsu/NewDisk/Garage/weights/ltp/base1.tgz \
    --bert=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/vocab.txt \
    --save_path=data/processed/pretrain-v0/ref.${data_type}.txt
done

# python prepare_corpus.py \
#     --version=pretrain-v0 \
#     --output_dir=data/processed/ \
#     --min_length=0 \
#     --max_length=256 \
#     --seed=42

# python run_chinese_ref.py \
#     --file_name=data/processed/pretrain-v0/corpus.txt \
#     --ltp=/home/louishsu/NewDisk/Garage/weights/ltp/base1.tgz \
#     --bert=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/vocab.txt \
#     --save_path=data/processed/pretrain-v0/ref.txt

export WANDB_DISABLED=true
data_dir=data/processed/pretrain-v0
version=nezha-cn-base-wwm-30k-mlm0.5
nohup python run_mlm_wwm.py \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --model_type=nezha \
    --train_file=${data_dir}/corpus.train.txt \
    --validation_file=${data_dir}/corpus.valid.txt \
    --train_ref_file=${data_dir}/ref.train.txt \
    --validation_ref_file=${data_dir}/ref.valid.txt \
    --cache_dir=cache/ \
    --overwrite_cache \
    --max_seq_length=256 \
    --preprocessing_num_workers=8 \
    --mlm_probability=0.5 \
    --output_dir=outputs/${version}/ \
    --overwrite_output_dir \
    --do_train --do_eval \
    --warmup_steps=1000 \
    --max_steps=30000 \
    --evaluation_strategy=steps \
    --eval_steps=1000 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --label_smoothing_factor=0.0 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --logging_dir=outputs/${version}/log/ \
    --logging_strategy=steps \
    --logging_steps=1000 \
    --save_strategy=steps \
    --save_steps=1000 \
    --save_total_limit=10 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --fp16 \
>> output/${version}.out &


version=pretrain-v1
python prepare_corpus.py \
    --version=${version} \
    --output_dir=data/processed/ \
    --min_length=0 \
    --max_length=128 \
    --train_ratio=0.9 \
    --seed=42

for data_type in train valid
do
python run_chinese_ref.py \
    --file_name=data/processed/${version}/corpus.${data_type}.txt \
    --ltp=/home/louishsu/NewDisk/Garage/weights/ltp/base1.tgz \
    --bert=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/vocab.txt \
    --seg_save_path=data/processed/${version}/seg.${data_type}.txt \
    --ref_save_path=data/processed/${version}/ref.${data_type}.txt
done

export WANDB_DISABLED=true
data_dir=data/processed/pretrain-v1
version=nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-50k-warmup3k-bs64x2
python run_mlm_wwm.py \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --output_dir=outputs/${version}/ \
    --do_train --do_eval \
    --warmup_steps=3000 \
    --max_steps=50000 \
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

export WANDB_DISABLED=true
data_dir=data/processed/pretrain-v1
version=nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2
python run_mlm_wwm.py \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --output_dir=outputs/${version}/ \
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

export WANDB_DISABLED=true
data_dir=data/processed/pretrain-v1
version=nezha-cn-large-wwm-seq128-lr3e-5-mlm0.15-100k-warmup10k-bs64x2
python run_mlm_wwm.py \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-large/ \
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
    --output_dir=outputs/${version}/ \
    --do_train --do_eval \
    --warmup_steps=10000 \
    --max_steps=100000 \
    --evaluation_strategy=steps \
    --eval_steps=5000 \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --label_smoothing_factor=0.0 \
    --learning_rate=3e-5 \
    --weight_decay=0.01 \
    --logging_dir=outputs/${version}/log/ \
    --logging_strategy=steps \
    --logging_steps=5000 \
    --save_strategy=steps \
    --save_steps=5000 \
    --save_total_limit=20 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --fp16

# 2022/4/11
## 以下路径为初始化权重
MODEL_NAME_OR_PATH=outputs/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000
SAVE_PATH=${MODEL_NAME_OR_PATH}-word
cp -r ${MODEL_NAME_OR_PATH} ${SAVE_PATH}    # 为了读取optimizer state等
python extend_chinese_ref_embeddings.py \
    --model_name_or_path=${MODEL_NAME_OR_PATH} \
    --model_type=nezha \
    --save_path=${SAVE_PATH}
# Add 0 tokens
# Model saved in outputs/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000-word
export WANDB_DISABLED=true
data_dir=data/processed/pretrain-v1
version=nezha-cn-base-wwm-word-seq128-lr2e-5-mlm0.15-200k-warmup3k-bs64x2
python run_mlm_wwm.py \
    --model_name_or_path=${SAVE_PATH} \
    --model_type=nezha \
    --train_file=${data_dir}/corpus.train.txt \
    --validation_file=${data_dir}/corpus.valid.txt \
    --train_ref_file=${data_dir}/ref.train.txt \
    --validation_ref_file=${data_dir}/ref.valid.txt \
    --do_ref_tokenize \
    --cache_dir=cache/ \
    --overwrite_cache \
    --max_seq_length=128 \
    --preprocessing_num_workers=8 \
    --mlm_probability=0.15 \
    --output_dir=outputs/${version}/ \
    --do_train --do_eval \
    --warmup_steps=3000 \
    --max_steps=200000 \
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

version=pretrain-v1
python prepare_corpus.py \
    --version=${version} \
    --output_dir=data/processed/ \
    --min_length=0 \
    --max_length=128 \
    --train_ratio=0.9 \
    --seed=42

# Stage2
# - 添加初赛B榜测试集数据（10k条）
# - 最短文本10，训练集比例0.95
# - 用ltp分词
# - MacBERT(n-gram + mlm_as_correction)
# - 调整超参数，无warmup
version=pretrain-v2
python prepare_corpus.py \
    --version=${version} \
    --output_dir=../data/tmp_data/ \
    --min_length=10 \
    --max_length=128 \
    --train_ratio=0.95 \
    --seed=42
for data_type in train valid
do
python run_chinese_ref.py \
    --file_name=../data/tmp_data/${version}/corpus.${data_type}.txt \
    --ltp=/home/louishsu/NewDisk/Garage/weights/ltp/base1.tgz \
    --bert=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/vocab.txt \
    --seg_save_path=../data/tmp_data/${version}/seg.${data_type}.txt \
    --ref_save_path=../data/tmp_data/${version}/ref.${data_type}.txt
done

# python generate_word_synonyms_map_from_tencent_ailab_embedding.py
# python convert_tencent_w2v_txt2bin.py
# export SYNONYMS_WORD2VEC_BIN_MODEL_ZH_CN="../data/public_data/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin"

export WANDB_DISABLED=true
data_dir=../data/tmp_data/pretrain-v2
version=nezha-cn-base-mac-seq128-lr2e-5-mlm0.15-4gram-100k-bs64x2
python run_mlm_wwm.py \
    --model_type=nezha \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --train_file=${data_dir}/corpus.train.txt \
    --validation_file=${data_dir}/corpus.valid.txt \
    --train_ref_file=${data_dir}/ref.train.txt \
    --validation_ref_file=${data_dir}/ref.valid.txt \
    --cache_dir=cache/ \
    --overwrite_cache \
    --max_seq_length=128 \
    --preprocessing_num_workers=8 \
    --mlm_probability=0.15 \
    --max_ngram=4 \
    --mlm_as_correction \
    --output_dir=../data/pretrain_model/${version}/ \
    --do_train --do_eval \
    --warmup_steps=0 \
    --max_steps=100000 \
    --evaluation_strategy=steps \
    --eval_steps=5000 \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --label_smoothing_factor=0.0 \
    --learning_rate=2e-5 \
    --weight_decay=0.01 \
    --logging_dir=../data/pretrain_model/${version}/log/ \
    --logging_strategy=steps \
    --logging_steps=5000 \
    --save_strategy=steps \
    --save_steps=5000 \
    --save_total_limit=20 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --fp16

export WANDB_DISABLED=true
data_dir=../data/tmp_data/pretrain-v2
version=nezha-cn-base-wwm-4gram-seq128-lr2e-5-mlm0.15-200k-warmup5k-bs64x2
python run_mlm_wwm.py \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --max_ngram=4 \
    --output_dir=../data/pretrain_model/${version}/ \
    --do_train --do_eval \
    --warmup_steps=5000 \
    --max_steps=200000 \
    --evaluation_strategy=steps \
    --eval_steps=5000 \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --label_smoothing_factor=0.0 \
    --learning_rate=2e-5 \
    --weight_decay=0.01 \
    --logging_dir=../data/pretrain_model/${version}/log/ \
    --logging_strategy=steps \
    --logging_steps=5000 \
    --save_strategy=steps \
    --save_steps=5000 \
    --save_total_limit=20 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --fp16
    
export WANDB_DISABLED=true
data_dir=../data/tmp_data/pretrain-v2
version=nezha-cn-base-wwm-4gram-seq128-lr3e-5-mlm0.15-100k-warmup1k-bs64x2
python run_mlm_wwm.py \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --max_ngram=4 \
    --output_dir=../data/pretrain_model/${version}/ \
    --do_train --do_eval \
    --warmup_steps=1000 \
    --max_steps=100000 \
    --evaluation_strategy=steps \
    --eval_steps=5000 \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --label_smoothing_factor=0.0 \
    --learning_rate=3e-5 \
    --weight_decay=0.01 \
    --logging_dir=../data/pretrain_model/${version}/log/ \
    --logging_strategy=steps \
    --logging_steps=5000 \
    --save_strategy=steps \
    --save_steps=5000 \
    --save_total_limit=20 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --fp16
    