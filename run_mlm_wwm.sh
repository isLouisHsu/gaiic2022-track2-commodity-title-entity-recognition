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
version=nezha-legal-cn-base-wwm-10k-mlm0.5
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
    --output_dir=output/${version}/ \
    --overwrite_output_dir \
    --do_train --do_eval \
    --warmup_steps=1000 \
    --max_steps=10000 \
    --evaluation_strategy=steps \
    --eval_steps=500 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --label_smoothing_factor=0.0 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --logging_dir=output/${version}/log/ \
    --logging_strategy=steps \
    --logging_steps=500 \
    --save_strategy=steps \
    --save_steps=500 \
    --save_total_limit=10 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --fp16 \
>> output/${version}.out &
