python run_span_classification_v1.py \
    --experiment_code=hfl-chinese-roberta-wwm-ext-span-v1-lr1e-5-wd0.01-dropout0.5-span15-e15-bs16x1-sinusoidal-biaffine \
    --task_name=gaiic \
    --model_type=bert \
    --pretrained_model_path=hfl/chinese-roberta-wwm-ext \
    --data_dir=data/raw/ \
    --train_input_file=train_500.txt \
    --eval_input_file=train_500.txt \
    --test_input_file=train_500.txt \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=256 \
    --eval_max_seq_length=256 \
    --test_max_seq_length=256 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=15 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.5 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --seed=42
2022-03-27 13:10:13 - INFO - root - ***** Evaluating results of gaiic *****
2022-03-27 13:10:13 - INFO - root -   global step = 375
2022-03-27 13:10:13 - INFO - root -   eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.7249  0.7182  0.7216     1629
1       0     0.0000  0.0000  0.0000        0
2       1     0.7200  0.9153  0.8060       59
3      10     0.4000  0.2857  0.3333       14
4      11     0.6928  0.7211  0.7067      147
5      12     0.8667  0.7027  0.7761       37
6      13     0.7293  0.6554  0.6904      148
7      14     0.8571  0.9296  0.8919       71
8      15     0.0000  0.0000  0.0000        2
9      16     0.8421  0.8649  0.8533       74
10     17     0.0000  0.0000  0.0000        0
11     18     0.6571  0.7931  0.7188      116
12     19     0.0000  0.0000  0.0000        1
13      2     1.0000  0.0909  0.1667       11
14     20     0.0000  0.0000  0.0000        2
15     21     0.0000  0.0000  0.0000        3
16     22     0.4286  0.1579  0.2308       19
17     23     0.0000  0.0000  0.0000        0
18     24     0.0000  0.0000  0.0000        0
19     25     0.0000  0.0000  0.0000        0
20     26     0.0000  0.0000  0.0000        0
21     27     0.0000  0.0000  0.0000        0
22     28     0.0000  0.0000  0.0000        0
23     29     0.6667  0.6667  0.6667        9
24      3     0.2500  0.2857  0.2667       21
25     30     0.0000  0.0000  0.0000        0
26     31     1.0000  0.5000  0.6667        2
27     32     0.0000  0.0000  0.0000        0
28     33     0.0000  0.0000  0.0000        0
29     34     0.0000  0.0000  0.0000        1
30     35     0.0000  0.0000  0.0000        0
31     36     0.8571  0.5000  0.6316       12
32     37     0.6154  0.5000  0.5517       32
33     38     0.6250  0.5147  0.5645       68
34     39     0.6000  0.1875  0.2857       16
35      4     0.7348  0.8366  0.7824      404
36     40     0.8000  0.6154  0.6957      104
37     41     0.0000  0.0000  0.0000        1
38     42     0.0000  0.0000  0.0000        0
39     43     0.0000  0.0000  0.0000        1
40     44     0.0000  0.0000  0.0000        0
41     45     0.0000  0.0000  0.0000        0
42     46     0.0000  0.0000  0.0000        0
43     47     0.0000  0.0000  0.0000        3
44     48     0.0000  0.0000  0.0000        0
45     49     0.0000  0.0000  0.0000        1
46      5     0.6975  0.7685  0.7313      108
47     50     0.0000  0.0000  0.0000        0
48     51     0.0000  0.0000  0.0000        0
49     52     0.0000  0.0000  0.0000        1
50     53     0.0000  0.0000  0.0000        0
51     54     0.7368  0.7368  0.7368       19
52      6     0.0000  0.0000  0.0000        6
53      7     0.8793  0.8644  0.8718       59
54      8     0.7143  0.8333  0.7692       36
55      9     0.8000  0.1905  0.3077       21
2022-03-27 13:10:13 - INFO - root -   eval_f1_micro_all_entity = 0.7216
2022-03-27 13:10:13 - INFO - root -   eval_loss = 0.0356


python prepare_data.py \
    --version=v0 \
    --labeled_files \
        data/raw/train_data/train.txt \
    --test_files \
        data/raw/preliminary_test_a/word_per_line_preliminary_A.txt \
    --output_dir=data/processed/ \
    --n_splits=5 \
    --shuffle \
    --seed=42
split=0, #train=32000, #dev=8000
split=1, #train=32000, #dev=8000
split=2, #train=32000, #dev=8000
split=3, #train=32000, #dev=8000
split=4, #train=32000, #dev=8000
test file: data/raw/preliminary_test_a/word_per_line_preliminary_A.txt, #test=10000

python run_span_classification_v1.py \
    --experiment_code=hfl-chinese-roberta-wwm-ext-span-v1-lr3e-5-wd0.01-dropout0.5-span15-e5-bs16x1-sinusoidal-biaffine \
    --task_name=gaiic \
    --model_type=bert \
    --pretrained_model_path=hfl/chinese-roberta-wwm-ext \
    --data_dir=data/processed/v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=256 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=5 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.5 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --seed=42
2022-03-27 14:32:11 - INFO - root - ***** Evaluating results of gaiic *****
2022-03-27 14:32:11 - INFO - root -   global step = 10000
2022-03-27 14:32:11 - INFO - root -   eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.7971  0.8095  0.8033   131361
1       1     0.8902  0.9530  0.9205     5061
2      10     0.6057  0.6140  0.6098     1671
3      11     0.7961  0.8139  0.8049    12120
4      12     0.8077  0.8505  0.8286     2469
5      13     0.7587  0.7409  0.7497    12729
6      14     0.8835  0.9220  0.9024     4385
7      15     0.6102  0.7770  0.6835      139
8      16     0.9125  0.9416  0.9269     4333
9      17     0.1667  0.2000  0.1818        5
10     18     0.7975  0.8220  0.8096    10700
11     19     0.2917  0.2414  0.2642       29
12      2     0.4084  0.2228  0.2883      570
13     20     0.4167  0.2586  0.3191      116
14     21     0.3165  0.5102  0.3906       98
15     22     0.4747  0.3652  0.4128     1851
16     23     0.0000  0.0000  0.0000        2
17     24     0.0000  0.0000  0.0000        1
18     25     0.3333  0.1000  0.1538       10
19     26     0.0000  0.0000  0.0000        0
20     28     0.0000  0.0000  0.0000        6
21     29     0.7168  0.7769  0.7457      883
22      3     0.6058  0.7106  0.6540     1821
23     30     0.3733  0.2393  0.2917      117
24     31     0.5091  0.3373  0.4058      166
25     32     0.3333  0.2000  0.2500        5
26     33     0.5000  0.5000  0.5000        2
27     34     0.1818  0.0488  0.0769       41
28     35     0.0000  0.0000  0.0000        0
29     36     0.6105  0.5614  0.5849      684
30     37     0.8206  0.8782  0.8484     3063
31     38     0.7200  0.7432  0.7314     6082
32     39     0.6005  0.4961  0.5433     1018
33      4     0.8525  0.8830  0.8675    33349
34     40     0.7346  0.7465  0.7405     6379
35     41     0.5000  0.1818  0.2667       88
36     42     0.0000  0.0000  0.0000        3
37     43     0.0000  0.0000  0.0000       13
38     44     0.2500  0.1667  0.2000        6
39     46     0.7500  0.3750  0.5000        8
40     47     0.3981  0.3245  0.3576      265
41     48     0.4583  0.3143  0.3729       35
42     49     0.5078  0.4456  0.4746      294
43      5     0.7705  0.7631  0.7668     8160
44     50     0.6026  0.6267  0.6144       75
45     51     1.0000  0.2500  0.4000        4
46     52     0.6429  0.4865  0.5538       37
47     53     0.0000  0.0000  0.0000        0
48     54     0.7125  0.7363  0.7242     1168
49      6     0.7455  0.7616  0.7534      323
50      7     0.8960  0.9073  0.9016     4986
51      8     0.9092  0.9161  0.9126     3421
52      9     0.5943  0.5798  0.5870     2570
2022-03-27 14:32:11 - INFO - root -   eval_f1_micro_all_entity = 0.8033
2022-03-27 14:32:11 - INFO - root -   eval_loss = 0.0212

# 线下 0.8067 线上 0.8017764761534591
python run_span_classification_v1.py \
    --experiment_code=nezha-cn-base-span-v1-lr3e-5-wd0.01-dropout0.5-span15-e5-bs16x1-sinusoidal-biaffine \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --data_dir=data/processed/v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=256 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=5 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.5 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --seed=42
022-03-27 15:23:27 - INFO - root - ***** Evaluating results of gaiic *****
2022-03-27 15:23:27 - INFO - root -   global step = 10000
2022-03-27 15:23:27 - INFO - root -   eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.8003  0.8132  0.8067   131361
1       1     0.9041  0.9463  0.9247     5061
2      10     0.6004  0.6565  0.6272     1671
3      11     0.7955  0.8210  0.8080    12120
4      12     0.8126  0.8534  0.8325     2469
5      13     0.7624  0.7454  0.7538    12729
6      14     0.8798  0.9261  0.9023     4385
7      15     0.6185  0.7698  0.6859      139
8      16     0.9188  0.9398  0.9292     4333
9      17     0.2857  0.4000  0.3333        5
10     18     0.8009  0.8245  0.8125    10700
11     19     0.2000  0.1724  0.1852       29
12      2     0.4000  0.2211  0.2847      570
13     20     0.4239  0.3362  0.3750      116
14     21     0.3125  0.5102  0.3876       98
15     22     0.4929  0.3739  0.4252     1851
16     23     0.0000  0.0000  0.0000        2
17     24     0.0000  0.0000  0.0000        1
18     25     0.0000  0.0000  0.0000       10
19     26     0.0000  0.0000  0.0000        0
20     28     0.0000  0.0000  0.0000        6
21     29     0.7263  0.7814  0.7529      883
22      3     0.6081  0.7057  0.6533     1821
23     30     0.4217  0.2991  0.3500      117
24     31     0.5607  0.3614  0.4396      166
25     32     0.0000  0.0000  0.0000        5
26     33     0.5000  0.5000  0.5000        2
27     34     0.1429  0.0488  0.0727       41
28     35     0.0000  0.0000  0.0000        0
29     36     0.5716  0.5658  0.5687      684
30     37     0.8244  0.8857  0.8540     3063
31     38     0.7234  0.7458  0.7345     6082
32     39     0.6160  0.4695  0.5329     1018
33      4     0.8590  0.8849  0.8717    33349
34     40     0.7337  0.7554  0.7444     6379
35     41     0.3784  0.1591  0.2240       88
36     42     0.0000  0.0000  0.0000        3
37     43     0.0000  0.0000  0.0000       13
38     44     0.2500  0.1667  0.2000        6
39     46     0.5000  0.2500  0.3333        8
40     47     0.4364  0.3887  0.4112      265
41     48     0.4815  0.3714  0.4194       35
42     49     0.5104  0.4184  0.4598      294
43      5     0.7707  0.7721  0.7714     8160
44     50     0.6081  0.6000  0.6040       75
45     51     1.0000  0.2500  0.4000        4
46     52     0.5926  0.4324  0.5000       37
47     53     0.0000  0.0000  0.0000        0
48     54     0.7283  0.7320  0.7301     1168
49      6     0.7453  0.7430  0.7442      323
50      7     0.9023  0.9146  0.9084     4986
51      8     0.9076  0.9161  0.9118     3421
52      9     0.5923  0.5833  0.5877     2570
2022-03-27 15:23:27 - INFO - root -   eval_f1_micro_all_entity = 0.8067
2022-03-27 15:23:27 - INFO - root -   eval_loss = 0.0212

# 线下 0.8098 线上 0.8043585754429128
python run_span_classification_v1.py \
    --experiment_code=nezha-cn-base-span-v1-lr3e-5-wd0.01-dropout0.5-span15-e5-bs16x1-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --data_dir=data/processed/v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=256 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=5 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.5 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --seed=42
2022-03-27 16:15:39 - INFO - root - ***** Evaluating results of gaiic *****
2022-03-27 16:15:39 - INFO - root -   global step = 10000
2022-03-27 16:15:39 - INFO - root -   eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.8034  0.8164  0.8098   131361
1       1     0.9039  0.9520  0.9273     5061
2      10     0.6221  0.6463  0.6340     1671
3      11     0.8023  0.8233  0.8127    12120
4      12     0.8081  0.8599  0.8332     2469
5      13     0.7651  0.7486  0.7568    12729
6      14     0.8868  0.9270  0.9065     4385
7      15     0.6316  0.7770  0.6968      139
8      16     0.9181  0.9418  0.9298     4333
9      17     0.5000  0.4000  0.4444        5
10     18     0.8011  0.8295  0.8151    10700
11     19     0.3333  0.2414  0.2800       29
12      2     0.4174  0.2439  0.3079      570
13     20     0.4400  0.2845  0.3455      116
14     21     0.2778  0.4592  0.3462       98
15     22     0.5011  0.3809  0.4328     1851
16     23     0.0000  0.0000  0.0000        2
17     24     0.0000  0.0000  0.0000        1
18     25     0.5000  0.1000  0.1667       10
19     26     0.0000  0.0000  0.0000        0
20     28     0.0000  0.0000  0.0000        6
21     29     0.7330  0.7803  0.7559      883
22      3     0.6120  0.7024  0.6541     1821
23     30     0.3973  0.2479  0.3053      117
24     31     0.5091  0.3373  0.4058      166
25     32     0.5000  0.2000  0.2857        5
26     33     0.5000  0.5000  0.5000        2
27     34     0.0909  0.0244  0.0385       41
28     35     0.0000  0.0000  0.0000        0
29     36     0.5896  0.5629  0.5759      684
30     37     0.8244  0.8890  0.8555     3063
31     38     0.7242  0.7473  0.7356     6082
32     39     0.6431  0.4833  0.5519     1018
33      4     0.8596  0.8884  0.8738    33349
34     40     0.7370  0.7587  0.7477     6379
35     41     0.4872  0.2159  0.2992       88
36     42     0.0000  0.0000  0.0000        3
37     43     0.0000  0.0000  0.0000       13
38     44     0.5000  0.3333  0.4000        6
39     46     0.3333  0.1250  0.1818        8
40     47     0.4560  0.3321  0.3843      265
41     48     0.5000  0.3143  0.3860       35
42     49     0.4942  0.4354  0.4629      294
43      5     0.7733  0.7795  0.7764     8160
44     50     0.6571  0.6133  0.6345       75
45     51     1.0000  0.2500  0.4000        4
46     52     0.6400  0.4324  0.5161       37
47     53     0.0000  0.0000  0.0000        0
48     54     0.7295  0.7320  0.7308     1168
49      6     0.7492  0.7307  0.7398      323
50      7     0.9038  0.9178  0.9107     4986
51      8     0.9134  0.9161  0.9148     3421
52      9     0.5965  0.5856  0.5910     2570
2022-03-27 16:15:39 - INFO - root -   eval_f1_micro_all_entity = 0.8098
2022-03-27 16:15:39 - INFO - root -   eval_loss = 0.02

# 线下 0.8076 线上 0.8028753143741536
python run_span_classification_v1.py \
    --experiment_code=nezha-cn-base-span-v1-lr3e-5-wd0.01-dropout0.1-span15-e10-bs16x1-sinusoidal-biaffine-fgm1.0-lsr0.1 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --data_dir=data/processed/v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=256 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=10 \
    --loss_type=lsr \
    --label_smoothing=0.1 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.1 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --seed=42
2022-03-27 18:00:56 - INFO - root - ***** Evaluating results of gaiic *****
2022-03-27 18:00:56 - INFO - root -   global step = 20000
2022-03-27 18:00:56 - INFO - root -   eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.8037  0.8115  0.8076   131361
1       1     0.8991  0.9524  0.9250     5061
2      10     0.6278  0.6188  0.6233     1671
3      11     0.7826  0.8354  0.8082    12120
4      12     0.8014  0.8562  0.8279     2469
5      13     0.7906  0.7111  0.7487    12729
6      14     0.8895  0.9195  0.9042     4385
7      15     0.5736  0.8129  0.6726      139
8      16     0.9171  0.9446  0.9307     4333
9      17     0.0000  0.0000  0.0000        5
10     18     0.7841  0.8419  0.8120    10700
11     19     0.3333  0.2414  0.2800       29
12      2     0.4272  0.2263  0.2959      570
13     20     0.6000  0.1293  0.2128      116
14     21     0.2604  0.4490  0.3296       98
15     22     0.4639  0.3960  0.4273     1851
16     23     0.0000  0.0000  0.0000        2
17     24     0.0000  0.0000  0.0000        1
18     25     0.0000  0.0000  0.0000       10
19     26     0.0000  0.0000  0.0000        0
20     28     0.0000  0.0000  0.0000        6
21     29     0.6960  0.7882  0.7392      883
22      3     0.6135  0.7062  0.6566     1821
23     30     0.4390  0.4615  0.4500      117
24     31     0.5392  0.3313  0.4104      166
25     32     0.0000  0.0000  0.0000        5
26     33     0.0000  0.0000  0.0000        2
27     34     0.1333  0.0488  0.0714       41
28     35     0.0000  0.0000  0.0000        0
29     36     0.6175  0.5146  0.5614      684
30     37     0.8250  0.8893  0.8559     3063
31     38     0.7332  0.7420  0.7376     6082
32     39     0.6101  0.5226  0.5630     1018
33      4     0.8662  0.8822  0.8741    33349
34     40     0.7336  0.7589  0.7460     6379
35     41     0.3529  0.0682  0.1143       88
36     42     0.0000  0.0000  0.0000        3
37     43     0.0000  0.0000  0.0000       13
38     44     1.0000  0.1667  0.2857        6
39     46     0.0000  0.0000  0.0000        8
40     47     0.4295  0.2415  0.3092      265
41     48     0.5500  0.3143  0.4000       35
42     49     0.5152  0.4626  0.4875      294
43      5     0.7781  0.7645  0.7712     8160
44     50     0.6471  0.5867  0.6154       75
45     51     1.0000  0.2500  0.4000        4
46     52     0.7143  0.4054  0.5172       37
47     53     0.0000  0.0000  0.0000        0
48     54     0.7413  0.7312  0.7362     1168
49      6     0.7301  0.7368  0.7334      323
50      7     0.9040  0.9182  0.9110     4986
51      8     0.9218  0.9094  0.9155     3421
52      9     0.5931  0.5860  0.5895     2570
2022-03-27 18:00:56 - INFO - root -   eval_f1_micro_all_entity = 0.8076
2022-03-27 18:00:56 - INFO - root -   eval_loss = 0.7262

TODO: scheduler

python run_span_classification_v1.py \
    --experiment_code=nezha-cn-base-30k-mlm0.5-span-v1-lr3e-5-wd0.01-dropout0.1-span15-e10-bs16x1-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=outputs/nezha-cn-base-wwm-30k-mlm0.5/checkpoint-30000/ \
    --data_dir=data/processed/v0/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=256 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_test_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=10 \
    --loss_type=ce \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.1 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --seed=42
2022-03-28 22:57:29 - INFO - root - ***** Evaluating results of gaiic *****
2022-03-28 22:57:29 - INFO - root -   global step = 20000
2022-03-28 22:57:29 - INFO - root -   eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.8023  0.8070  0.8047   131361
1       1     0.9148  0.9467  0.9305     5061
2      10     0.6136  0.6367  0.6250     1671
3      11     0.8249  0.7867  0.8054    12120
4      12     0.8228  0.8254  0.8241     2469
5      13     0.7930  0.7070  0.7475    12729
6      14     0.8719  0.9300  0.9000     4385
7      15     0.5956  0.7842  0.6770      139
8      16     0.9140  0.9444  0.9289     4333
9      17     1.0000  0.2000  0.3333        5
10     18     0.7800  0.8454  0.8114    10700
11     19     0.3333  0.1379  0.1951       29
12      2     0.4245  0.2070  0.2783      570
13     20     0.3750  0.3621  0.3684      116
14     21     0.4844  0.3163  0.3827       98
15     22     0.4341  0.4360  0.4350     1851
16     23     0.0000  0.0000  0.0000        2
17     24     0.0000  0.0000  0.0000        1
18     25     0.0000  0.0000  0.0000       10
19     26     0.0000  0.0000  0.0000        0
20     28     0.1250  0.1667  0.1429        6
21     29     0.7611  0.7395  0.7501      883
22      3     0.6596  0.5958  0.6261     1821
23     30     0.3981  0.3504  0.3727      117
24     31     0.3548  0.3313  0.3427      166
25     32     0.5000  0.2000  0.2857        5
26     33     1.0000  0.5000  0.6667        2
27     34     0.1667  0.0488  0.0755       41
28     35     0.0000  0.0000  0.0000        0
29     36     0.5929  0.4152  0.4884      684
30     37     0.8083  0.8945  0.8492     3063
31     38     0.6950  0.7567  0.7245     6082
32     39     0.6396  0.3939  0.4875     1018
33      4     0.8460  0.9022  0.8732    33349
34     40     0.7937  0.6996  0.7437     6379
35     41     0.2857  0.0227  0.0421       88
36     42     0.0000  0.0000  0.0000        3
37     43     0.0000  0.0000  0.0000       13
38     44     0.0000  0.0000  0.0000        6
39     46     0.0000  0.0000  0.0000        8
40     47     0.4138  0.3623  0.3863      265
41     48     0.4722  0.4857  0.4789       35
42     49     0.4540  0.5034  0.4774      294
43      5     0.7531  0.7866  0.7695     8160
44     50     0.6462  0.5600  0.6000       75
45     51     1.0000  0.5000  0.6667        4
46     52     0.6207  0.4865  0.5455       37
47     53     0.0000  0.0000  0.0000        0
48     54     0.8044  0.6301  0.7067     1168
49      6     0.7904  0.7121  0.7492      323
50      7     0.9057  0.9089  0.9073     4986
51      8     0.9137  0.9068  0.9102     3421
52      9     0.5755  0.5813  0.5784     2570
2022-03-28 22:57:29 - INFO - root -   eval_df_without_label_entity = 
   label  precision  recall      f1  support
0  micro     0.8973  0.9026  0.8999   131361
1      _     0.8973  0.9026  0.8999   131361
2022-03-28 22:57:29 - INFO - root -   eval_f1_micro_all_entity = 0.8047
2022-03-28 22:57:29 - INFO - root -   eval_f1_micro_without_label_entity = 0.8999
2022-03-28 22:57:29 - INFO - root -   eval_loss = 0.0208

# FIXED：该函数无法提取由"B-X"标记的单个token实体
# 由5折调整为10折，且不打乱
python prepare_data.py \
    --version=v1 \
    --labeled_files \
        data/raw/train_data/train.txt \
    --test_files \
        data/raw/preliminary_test_a/word_per_line_preliminary_A.txt \
    --output_dir=data/processed/ \
    --n_splits=10 \
    --seed=42
split=0, #train=36000, #dev=4000
split=1, #train=36000, #dev=4000
split=2, #train=36000, #dev=4000
split=3, #train=36000, #dev=4000
split=4, #train=36000, #dev=4000
split=5, #train=36000, #dev=4000
split=6, #train=36000, #dev=4000
split=7, #train=36000, #dev=4000
split=8, #train=36000, #dev=4000
split=9, #train=36000, #dev=4000
test file: data/raw/preliminary_test_a/word_per_line_preliminary_A.txt, #test=10000

python run_span_classification_v1.py \
    --experiment_code=uer_large-span-v1-lr2e-5-wd0.01-dropout0.1-span15-e6-bs8x2-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=bert \
    --pretrained_model_path=junnyu/uer_large \
    --data_dir=data/processed/v1/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_eval \
    --evaluate_during_training \
    --train_max_seq_length=256 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_test_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=2e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.1 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --seed=42

TODO: 
1. 尝试[dbiir/UER-py](https://github.com/dbiir/UER-py)下其他模型，如`MixedCorpus+BertEncoder(xlarge)+MlmTarget`,`MixedCorpus+BertEncoder(xlarge)+BertTarget(WWM)`
2. 模型部署时，可参考[ELS-RD/transformer-deploy](https://github.com/ELS-RD/transformer-deploy)
