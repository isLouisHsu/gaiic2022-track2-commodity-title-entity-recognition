import os
import sys
import glob
import json
# sys.path.append("code/TorchBlocks/")
# from torchblocks.utils.options import Argparser
# from torchblocks.core.utils import is_apex_available
sys.path.append("code/")
from packages import Argparser, is_apex_available

# WARNING: 统一采用绝对路径！！
def pred_BIO(path_word: str, path_sample: str, batch_size: int = 1, 
    # model_path="/home/mw/project/data/best_model/gmodel_spancls_20220505065060",
    model_path="/home/mw/project/data/best_model/gmodel_spancls_202205077213",
    submit_result_file="/home/mw/project/results.txt"
):
    basename, ext = os.path.splitext(os.path.basename(path_word))
    json_file = glob.glob(os.path.join(model_path, "*_opts.json"))[0]
    opts = Argparser.parse_args_from_json(json_file=json_file)
    model_path = os.path.abspath(model_path)
    opts.output_dir = model_path
    opts.pretrained_model_path = model_path
    opts.checkpoint_predict_code = ""   # 线上只有一级目录
    opts.do_train = False
    opts.do_eval = False
    opts.do_predict = True
    opts.per_gpu_test_batch_size = batch_size
    opts.gradient_accumulation_steps = 1
    opts.data_dir = "../data/tmp_data/predict/"
    opts.test_input_file = f"{basename}.jsonl"
    opts.fp16= False
    json_file = os.path.join(model_path, "predict_opts.json")
    with open(str(json_file), 'w') as f:
        json.dump(vars(opts), f, ensure_ascii=False, indent=4)

    # if not is_apex_available():
    #     cmd = \
    #         """
    #         # git clone https://github.com/NVIDIA/apex
    #         cd apex
    #         pip install -v --disable-pip-version-check --no-cache-dir ./
    #         """
    #     print(cmd)
    #     os.system(cmd)

    path_word = os.path.abspath(path_word)
    cmd = \
        """
        # sh init.sh
        cd code/
        python prepare_data.py \
            --version=predict \
            --labeled_files \
                ../data/contest_data/train_data/train.txt \
            --test_files %s \
            --output_dir=../data/tmp_data/ \
            --n_splits=1 \
            --seed=42
        python run_span_classification_v1.py \
            %s
        """ % (path_word, json_file)
    print(cmd)
    os.system(cmd)
    
    checkpoint_path = os.path.join(model_path, opts.checkpoint_predict_code)
    result_file_path = glob.glob(os.path.join(checkpoint_path, "*.predictions.txt"))[0]
    cmd = \
        """
        cp %s %s
        """ % (result_file_path, submit_result_file)
    print(cmd)
    os.system(cmd)

# if __name__ == "__main__":
#     pred_BIO(
#         "data/contest_data/preliminary_test_b/word_per_line_preliminary_B.txt", 
#         "data/contest_data/preliminary_test_b/sample_per_line_preliminary_B.txt",
#         submit_result_file="results.txt",
#     )
    