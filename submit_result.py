import os
import sys
import glob
import json
sys.path.append("code/TorchBlocks/")
from torchblocks.utils.options import Argparser
from torchblocks.core.utils import is_apex_available

def pred_BIO(path_word: str, path_sample: str, batch_size: int = 1, 
    model_path="data/best_model/gaiic_nezha_nezha-4gram-200k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs8x2-sinusoidal-biaffine-fgm1.0-rdrop0.3",
    submit_result_file="/home/mw/project/results.txt"
):
    json_file = glob.glob(os.path.join(model_path, "*_opts.json"))[0]
    opts = Argparser.parse_args_from_json(json_file=json_file)
    opts.output_dir = os.path.join("../", model_path)
    opts.do_train = False
    opts.do_eval = False
    opts.do_predict = True
    opts.per_gpu_test_batch_size = batch_size
    opts.gradient_accumulation_steps = 1
    # opts.fp16= False
    json_file = os.path.join(model_path, "predict_opts.json")
    with open(str(json_file), 'w') as f:
        json.dump(vars(opts), f, ensure_ascii=False, indent=4)

    if not is_apex_available():
        cmd = \
            """
            git clone https://github.com/NVIDIA/apex
            cd apex
            pip install -v --disable-pip-version-check --no-cache-dir ./
            """
        os.system(cmd)

    cmd = \
        """
        # sh init.sh
        cd code/
        test_file=../%s
        python prepare_data.py \
            --version=predict \
            --labeled_files \
                ../data/contest_data/train_data/train.txt \
            --test_files \
                ${test_file} \
            --output_dir=../data/tmp_data/ \
            --n_splits=1 \
            --seed=42
        python run_span_classification_v1.py \
            ../%s
        """ % (path_word, json_file)
    os.system(cmd)
    checkpoint_path = os.path.join(model_path, opts.checkpoint_predict_code)
    result_file_path = glob.glob(os.path.join(checkpoint_path, "*.predictions.txt"))[0]
    cmd = \
        """
        cp %s %s
        """ % (result_file_path, submit_result_file)
    os.system(cmd)

# if __name__ == "__main__":
#     pred_BIO(
#         "data/contest_data/preliminary_test_b/word_per_line_preliminary_B.txt", 
#         "data/contest_data/preliminary_test_b/sample_per_line_preliminary_B.txt",
#         submit_result_file="../results.txt",
#     )
    