# import os
# import sys
# import glob
# import json
# # sys.path.append("code/TorchBlocks/")
# # from torchblocks.utils.options import Argparser
# # from torchblocks.core.utils import is_apex_available
# sys.path.append("code/")
# from packages import Argparser, is_apex_available

# # WARNING: 统一采用绝对路径！！
# def pred_BIO(path_word: str, path_sample: str, batch_size: int = 1, 
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_20220505065060", # 0.8105264749920598
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_202205077213",   # 0.8132664117368327
#     # model_path="/home/mw/project/data/best_model/gmodel_gp_20220508xxxx",        # 0.8135973553679232
#     # model_path="/home/mw/project/data/best_model/gmodel_gpswa_202205089550",     # 0.8135456792962686
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_20220509016006", # 0.8131485592934022
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_202205101483",   # 0.8144165048602052
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_2022051219618",  # 0.8154596708823378
#     # model_path="/home/mw/project/data/best_model/gmodel_gp_2022051235705",       # 0.814864726037745
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_2022051428128",  # 0.8153485305142973
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_2022051515130",  # 0.8153156273245536
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_2022051528877",  # 0.8154189358838884
#     # model_path="/home/mw/project/data/best_model/gmodel_spancls_2022051617829",  # 0.8104775463912551
#     model_path="/home/mw/project/data/best_model/gmodel_spancls_2022051629391",  # 0.815385819848358
#     submit_result_file="/home/mw/project/results.txt"
# ):
#     basename, ext = os.path.splitext(os.path.basename(path_word))
#     model_path = os.path.abspath(model_path)
#     print(model_path); os.system("ls %s" % model_path)
#     json_file = glob.glob(os.path.join(model_path, "*_opts.json"))[0]
#     opts = Argparser.parse_args_from_json(json_file=json_file)
#     opts.output_dir = model_path
#     opts.pretrained_model_path = model_path
#     opts.checkpoint_predict_code = ""   # 线上只有一级目录
#     opts.do_train = False
#     opts.do_eval = False
#     opts.do_predict = True
#     opts.per_gpu_test_batch_size = batch_size
#     opts.gradient_accumulation_steps = 1
#     opts.data_dir = "../data/tmp_data/predict/"
#     opts.test_input_file = f"{basename}.jsonl"
#     opts.fp16= False
#     json_file = os.path.join(model_path, "predict_opts.json")
#     with open(str(json_file), 'w') as f:
#         json.dump(vars(opts), f, ensure_ascii=False, indent=4)

#     # if not is_apex_available():
#     #     cmd = \
#     #         """
#     #         # git clone https://github.com/NVIDIA/apex
#     #         cd apex
#     #         pip install -v --disable-pip-version-check --no-cache-dir ./
#     #         """
#     #     print(cmd)
#     #     os.system(cmd)

#     path_word = os.path.abspath(path_word)
#     cmd = \
#         """
#         # sh init.sh
#         cd code/
#         python prepare_data.py \
#             --version=predict \
#             --labeled_files \
#                 ../data/contest_data/train_data/train.txt \
#             --test_files %s \
#             --output_dir=../data/tmp_data/ \
#             --n_splits=1 \
#             --seed=42
#         python run_span_classification_v1.py \
#             %s
#         """ % (path_word, json_file)
#     print(cmd)
#     os.system(cmd)
    
#     checkpoint_path = os.path.join(model_path, opts.checkpoint_predict_code)
#     result_file_path = glob.glob(os.path.join(checkpoint_path, "*.predictions.txt"))[0]
#     cmd = \
#         """
#         cp %s %s
#         """ % (result_file_path, submit_result_file)
#     print(cmd)
#     os.system(cmd)

# # if __name__ == "__main__":
# #     pred_BIO(
# #         "data/contest_data/preliminary_test_b/word_per_line_preliminary_B.txt", 
# #         "data/contest_data/preliminary_test_b/sample_per_line_preliminary_B.txt",
# #         submit_result_file="results.txt",
# #     )

# ---------------------------------------------------------------------------------------------
import os
import glob
import json

# WARNING: 统一采用绝对路径！！
def pred_BIO(path_word: str, path_sample: str, batch_size: int = 1,
    # model_path="/home/mw/project/best_model",
    model_path="/home/mw/project/data/best_model/gmodel_gpv2_2022051811505", # 0.8176488551464742
    submit_result_file="/home/mw/project/results.txt"
):
    model_path = os.path.abspath(model_path)
    print(model_path); os.system("ls %s" % model_path)
    # cmd = \
    #     f"""
    #     python exp_gaiic_global_pointer_v2.py \
    #         --experiment_code=experiment_bert_base_fold0_gp_v2_pre_v62 \
    #         --task_name=gaiic \
    #         --model_type=nezha \
    #         --do_lower_case \
    #         --pretrained_model_path={model_path} \
    #         --data_dir=./ \
    #         --output_dir=./ \
    #         --do_predict_test \
    #         --test_input_file={path_sample} \
    #         --eval_checkpoint_path={model_path} \
    #         --submit_file_path={submit_result_file} \
    #         --evaluate_during_training \
    #         --train_max_seq_length=128 \
    #         --eval_max_seq_length=128 \
    #         --test_max_seq_length=128 \
    #         --per_gpu_train_batch_size=16 \
    #         --per_gpu_eval_batch_size=32 \
    #         --per_gpu_test_batch_size={batch_size} \
    #         --gradient_accumulation_steps=1 \
    #         --learning_rate=3e-5 \
    #         --other_learning_rate=1e-3 \
    #         --weight_decay=0.001 \
    #         --scheduler_type=cosine \
    #         --base_model_name=bert \
    #         --warmup_proportion=0.1 \
    #         --max_grad_norm=1.0 \
    #         --num_train_epochs=10 \
    #         --use_rope \
    #         --do_lstm \
    #         --num_lstm_layers=2 \
    #         --adam_epsilon=1e-8 \
    #         --post_lstm_dropout=0.5 \
    #         --inner_dim=64 \
    #         --loss_type=pcl \
    #         --do_awp \
    #         --do_rdrop \
    #         --seed=42
    #     """
    cmd = \
        f"""
        cd code/
        python exp_gaiic_global_pointer_v2.py \
            --experiment_code=experiment_bert_base_fold0_gp_v2_pre_v62 \
            --task_name=gaiic \
            --model_type=nezha \
            --do_lower_case \
            --pretrained_model_path={model_path} \
            --data_dir=/home/mw/temp/10_folds_data/ \
            --train_input_file=train.all.jsonl \
            --eval_input_file=dev.0.jsonl \
            --output_dir=../data/model_data/ \
            --do_predict_test \
            --save_best \
            --test_input_file={path_sample} \
            --eval_checkpoint_path={model_path} \
            --submit_file_path={submit_result_file} \
            --evaluate_during_training \
            --train_max_seq_length=128 \
            --eval_max_seq_length=128 \
            --test_max_seq_length=128 \
            --per_gpu_train_batch_size=16 \
            --per_gpu_eval_batch_size=32 \
            --per_gpu_test_batch_size={batch_size} \
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
        """
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    pred_BIO(
         path_word= "",
         path_sample="./test_submit_dev_0.txt",
         model_path='./best_model',
         submit_result_file="tmp_results.txt",
     )
    dev_0 = './dev.0.jsonl'
    import json
    from utils import get_entity_biob
    true = []
    with open(dev_0) as fr:
     for line in fr.readlines():
         line = json.loads(line)
         true.append(line)
    sentences = []
    sentence_counter = 0
    with open('tmp_results.txt', encoding="utf-8") as f:
        lines = f.readlines()
    current_words = []
    current_labels = []
    for row in lines:
        row = row.rstrip("\n")
        if row != "":
            token, label = row[0], row[2:]
            current_words.append(token)
            current_labels.append(label)
        else:
            if not current_words:
                continue
            assert len(current_words) == len(current_labels), "word len doesn't match label length"
            sentence = {
                "id": str(sentence_counter),
                "tokens": current_words,
                "ner_tags": current_labels
            }
            sentence_counter += 1
            current_words = []
            current_labels = []
            sentences.append(sentence)
    result = []
    for x,y in zip(true,sentences):
        true_ents = get_entity_biob(x['ner_tags'],None)
        pred_ents = get_entity_biob(y['ner_tags'],None)
        x['true'] = true_ents
        x['pred'] = pred_ents
        assert "".join(x['tokens']) == "".join(y['tokens'])
        result.append(x)
    import torch
    import numpy as np
    class MetricsCalculator(object):
        def __init__(self):
            super().__init__()

        def get_sample_f1(self, y_pred, y_true):
            y_pred = torch.gt(y_pred, 0).float()
            return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

        def get_sample_precision(self, y_pred, y_true):
            y_pred = torch.gt(y_pred, 0).float()
            return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

        def get_evaluate_fpr(self, y_pred):
            pred = []
            true = []
            for i, x in enumerate(y_pred):
                p = x['pred']
                for pp in p:
                    pred.append((i, pp[0], pp[1], pp[2]))
                t = x['true']
                for tt in t:
                    true.append((i, tt[0], tt[1], tt[2]))
            R = set(pred)
            T = set(true)
            X = len(R & T)
            Y = len(R)
            Z = len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            return f1, precision, recall

    m = MetricsCalculator()
    m.get_evaluate_fpr(result)
