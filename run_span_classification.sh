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
    --do_eval \
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

# drop_overlap_baseline
***** Evaluating results of gaiic *****
  global step = 0
  eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.8072  0.8112  0.8092   131361
1       1     0.9055  0.9506  0.9275     5061
2      10     0.6289  0.6379  0.6334     1671
3      11     0.8039  0.8182  0.8110    12120
4      12     0.8119  0.8546  0.8327     2469
5      13     0.7671  0.7462  0.7565    12729
6      14     0.8891  0.9250  0.9067     4385
7      15     0.6463  0.7626  0.6997      139
8      16     0.9210  0.9384  0.9296     4333
9      17     0.3333  0.2000  0.2500        5
10     18     0.8058  0.8212  0.8134    10700
11     19     0.3333  0.2414  0.2800       29
12      2     0.4277  0.2386  0.3063      570
13     20     0.4521  0.2845  0.3492      116
14     21     0.2763  0.4286  0.3360       98
15     22     0.5025  0.3760  0.4302     1851
16     23     0.0000  0.0000  0.0000        2
17     24     0.0000  0.0000  0.0000        1
18     25     0.5000  0.1000  0.1667       10
19     26     0.0000  0.0000  0.0000        0
20     28     0.0000  0.0000  0.0000        6
21     29     0.7341  0.7724  0.7528      883
22      3     0.6152  0.6980  0.6540     1821
23     30     0.4000  0.2393  0.2995      117
24     31     0.5093  0.3313  0.4015      166
25     32     0.5000  0.2000  0.2857        5
26     33     0.5000  0.5000  0.5000        2
27     34     0.0909  0.0244  0.0385       41
28     35     0.0000  0.0000  0.0000        0
29     36     0.5880  0.5570  0.5721      684
30     37     0.8349  0.8802  0.8570     3063
31     38     0.7342  0.7348  0.7345     6082
32     39     0.6575  0.4695  0.5479     1018
33      4     0.8624  0.8838  0.8729    33349
34     40     0.7416  0.7526  0.7471     6379
35     41     0.4872  0.2159  0.2992       88
36     42     0.0000  0.0000  0.0000        3
37     43     0.0000  0.0000  0.0000       13
38     44     0.5000  0.3333  0.4000        6
39     46     0.3333  0.1250  0.1818        8
40     47     0.4703  0.3283  0.3867      265
41     48     0.4762  0.2857  0.3571       35
42     49     0.5080  0.4320  0.4669      294
43      5     0.7759  0.7732  0.7745     8160
44     50     0.6571  0.6133  0.6345       75
45     51     1.0000  0.2500  0.4000        4
46     52     0.6400  0.4324  0.5161       37
47     53     0.0000  0.0000  0.0000        0
48     54     0.7317  0.7286  0.7302     1168
49      6     0.7622  0.7245  0.7429      323
50      7     0.9058  0.9162  0.9110     4986
51      8     0.9138  0.9144  0.9141     3421
52      9     0.5992  0.5805  0.5897     2570
  eval_df_without_label_entity = 
   label  precision  recall      f1  support
0  micro        0.9  0.9044  0.9022   131361
1      _        0.9  0.9044  0.9022   131361
  eval_f1_micro_all_entity = 0.8092
  eval_f1_micro_without_label_entity = 0.9022
  eval_loss = 0.02

# drop_overlap_nms
***** Evaluating results of gaiic *****
  global step = 0
  eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.8071  0.8124  0.8097   131361
1       1     0.9050  0.9506  0.9272     5061
2      10     0.6256  0.6349  0.6302     1671
3      11     0.8049  0.8203  0.8125    12120
4      12     0.8103  0.8562  0.8326     2469
5      13     0.7667  0.7467  0.7566    12729
6      14     0.8885  0.9254  0.9066     4385
7      15     0.6228  0.7482  0.6797      139
8      16     0.9214  0.9391  0.9302     4333
9      17     0.5000  0.4000  0.4444        5
10     18     0.8066  0.8234  0.8149    10700
11     19     0.3333  0.2414  0.2800       29
12      2     0.4259  0.2368  0.3044      570
13     20     0.4521  0.2845  0.3492      116
14     21     0.2745  0.4286  0.3347       98
15     22     0.5025  0.3793  0.4323     1851
16     23     0.0000  0.0000  0.0000        2
17     24     0.0000  0.0000  0.0000        1
18     25     0.5000  0.1000  0.1667       10
19     26     0.0000  0.0000  0.0000        0
20     28     0.0000  0.0000  0.0000        6
21     29     0.7355  0.7746  0.7546      883
22      3     0.6125  0.6969  0.6519     1821
23     30     0.4028  0.2479  0.3069      117
24     31     0.5138  0.3373  0.4073      166
25     32     0.5000  0.2000  0.2857        5
26     33     0.5000  0.5000  0.5000        2
27     34     0.0909  0.0244  0.0385       41
28     35     0.0000  0.0000  0.0000        0
29     36     0.5908  0.5614  0.5757      684
30     37     0.8310  0.8861  0.8576     3063
31     38     0.7334  0.7354  0.7344     6082
32     39     0.6499  0.4705  0.5459     1018
33      4     0.8636  0.8843  0.8738    33349
34     40     0.7401  0.7545  0.7472     6379
35     41     0.4872  0.2159  0.2992       88
36     42     0.0000  0.0000  0.0000        3
37     43     0.0000  0.0000  0.0000       13
38     44     0.5000  0.3333  0.4000        6
39     46     0.3333  0.1250  0.1818        8
40     47     0.4632  0.3321  0.3868      265
41     48     0.5000  0.3143  0.3860       35
42     49     0.5000  0.4286  0.4615      294
43      5     0.7760  0.7760  0.7760     8160
44     50     0.6571  0.6133  0.6345       75
45     51     1.0000  0.2500  0.4000        4
46     52     0.6400  0.4324  0.5161       37
47     53     0.0000  0.0000  0.0000        0
48     54     0.7317  0.7286  0.7302     1168
49      6     0.7597  0.7245  0.7417      323
50      7     0.9053  0.9168  0.9110     4986
51      8     0.9141  0.9146  0.9144     3421
52      9     0.5987  0.5840  0.5913     2570
  eval_df_without_label_entity = 
   label  precision  recall      f1  support
0  micro        0.9  0.9059  0.9029   131361
1      _        0.9  0.9059  0.9029   131361
  eval_f1_micro_all_entity = 0.8097
  eval_f1_micro_without_label_entity = 0.9029
  eval_loss = 0.02


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

python prepare_data.py \
    --version=v2 \
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
    --experiment_code=nezha-cn-base-spanv1-datav2-lr3e-5-wd0.01-dropout0.5-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --data_dir=data/processed/v2/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_predict \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_test_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_save_best \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.5 \
    --negative_sampling=0.0 \
    --max_span_length=35 \
    --width_embedding_size=64 \
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --seed=42
2022-03-30 23:47:56 - INFO - root - ***** Evaluating results of gaiic *****
2022-03-30 23:47:56 - INFO - root -   global step = 6000
2022-03-30 23:47:56 - INFO - root -   eval_df_all_entity = 
    label  precision  recall      f1  support
0   micro     0.8046  0.8113  0.8079   132430
1       1     0.8975  0.9510  0.9235     5062
2      10     0.6619  0.5834  0.6202     1678
3      11     0.7854  0.8307  0.8074    12128
4      12     0.8312  0.8335  0.8323     2522
5      13     0.7669  0.7382  0.7523    12871
6      14     0.8865  0.9207  0.9033     4429
7      15     0.6167  0.7986  0.6959      139
8      16     0.9220  0.9350  0.9285     4567
9      17     0.3333  0.6000  0.4286        5
10     18     0.7980  0.8298  0.8136    10764
11     19     0.2917  0.2414  0.2642       29
12      2     0.4154  0.2360  0.3010      572
13     20     0.6250  0.1724  0.2703      116
14     21     0.2866  0.4592  0.3529       98
15     22     0.5100  0.3276  0.3990     1874
16     23     0.0000  0.0000  0.0000        2
17     24     0.0000  0.0000  0.0000        1
18     25     0.0000  0.0000  0.0000       10
19     26     0.0000  0.0000  0.0000        0
20     28     0.0000  0.0000  0.0000        6
21     29     0.7300  0.7837  0.7559      883
22      3     0.6385  0.6605  0.6493     1829
23     30     0.4052  0.5299  0.4593      117
24     31     0.5667  0.3036  0.3953      168
25     32     0.0000  0.0000  0.0000        6
26     33     0.0000  0.0000  0.0000        2
27     34     0.1538  0.0889  0.1127       45
28     35     0.0000  0.0000  0.0000        1
29     36     0.6034  0.5548  0.5781      694
30     37     0.8195  0.8926  0.8545     3063
31     38     0.7244  0.7333  0.7288     6176
32     39     0.6155  0.5059  0.5553     1022
33      4     0.8615  0.8852  0.8732    33435
34     40     0.7388  0.7538  0.7463     6406
35     41     0.4000  0.0455  0.0816       88
36     42     0.0000  0.0000  0.0000        3
37     43     0.0000  0.0000  0.0000       13
38     44     0.0000  0.0000  0.0000        6
39     46     1.0000  0.1250  0.2222        8
40     47     0.4793  0.2189  0.3005      265
41     48     0.4737  0.2571  0.3333       35
42     49     0.4881  0.4881  0.4881      295
43      5     0.7914  0.7654  0.7782     8160
44     50     0.5897  0.5974  0.5935       77
45     51     1.0000  0.2500  0.4000        4
46     52     0.6522  0.3846  0.4839       39
47     53     0.0000  0.0000  0.0000        0
48     54     0.7196  0.7392  0.7293     1177
49      6     0.7554  0.7623  0.7588      324
50      7     0.9077  0.9150  0.9114     4989
51      8     0.9070  0.9159  0.9114     3651
52      9     0.5652  0.6223  0.5924     2576
2022-03-30 23:47:56 - INFO - root -   eval_df_without_label_entity = 
   label  precision  recall      f1  support
0  micro     0.8977  0.9052  0.9014   132430
1      _     0.8977  0.9052  0.9014   132430
2022-03-30 23:47:56 - INFO - root -   eval_f1_micro_all_entity = 0.8079
2022-03-30 23:47:56 - INFO - root -   eval_f1_micro_without_label_entity = 0.9014
2022-03-30 23:47:56 - INFO - root -   eval_loss = 0.011

# 线上 0.8041638451190515
python run_span_classification_v1.py \
    --experiment_code=nezha-cn-base-spanv1-datav2-lr5e-5-wd0.01-dropout0.1-span35-e15-bs32x1-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --data_dir=data/processed/v2/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_eval --do_predict \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_test_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=15 \
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
    --seed=42
2022-04-02 23:50:55 - INFO - root -   eval_f1_micro_all_entity = 0.8045
2022-04-02 23:50:55 - INFO - root -   eval_f1_micro_without_label_entity = 0.8976
2022-04-02 23:50:55 - INFO - root -   eval_loss = 0.0112

python run_span_classification_v1.py \
    --experiment_code=nezha-cn-base-spanv1-datav2-lr3e-5-wd0.01-dropout0.1-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --data_dir=data/processed/v2/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_test_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
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
    --seed=42
2022-04-03 17:31:34 - INFO - root -   eval_f1_micro_all_entity = 0.808
2022-04-03 17:31:34 - INFO - root -   eval_f1_micro_without_label_entity = 0.9007
2022-04-03 17:31:34 - INFO - root -   eval_loss = 0.0111

# 线上 0.809930004779276
python run_span_classification_v1.py \
    --experiment_code=nezha-50k-spanv1-datav2-lr3e-5-wd0.01-dropout0.1-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Code/gaiic2022-track2-commodity-title-entity-recognition/outputs/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-50k-warmup30k-bs64x2/checkpoint-50000/ \
    --data_dir=data/processed/v2/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_predict \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_test_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
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
    --seed=42
# eval_f1_micro_all_entity = 0.8093
# eval_f1_micro_without_label_entity = 0.9016
# eval_loss = 0.011

# 线上0.8108433916489609
python run_span_classification_v1.py \
    --experiment_code=nezha-50k-spanv1-datav2-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Code/gaiic2022-track2-commodity-title-entity-recognition/outputs/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-50k-warmup30k-bs64x2/checkpoint-50000/ \
    --data_dir=data/processed/v2/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_train --do_predict \
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
    --label_smoothing=0.0 \
    --decode_thresh=0.0 \
    --use_sinusoidal_width_embedding \
    --do_biaffine \
    --adv_enable \
    --adv_epsilon=1.0 \
    --do_rdrop \
    --rdrop_weight=0.3 \
    --seed=42
# eval_f1_micro_all_entity = 0.8098
# eval_f1_micro_without_label_entity = 0.9025
# eval_loss = 0.0109

python run_span_classification_v1.py \
    --experiment_code=nezha-50k-spanv1-datav2-augv1-lr3e-5-wd0.01-dropout0.1-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0 \
    --task_name=gaiic \
    --model_type=nezha \
    --pretrained_model_path=/home/louishsu/NewDisk/Code/gaiic2022-track2-commodity-title-entity-recognition/outputs/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-50k-warmup30k-bs64x2/checkpoint-50000/ \
    --data_dir=data/processed/v2/ \
    --train_input_file=train.0.jsonl \
    --eval_input_file=dev.0.jsonl \
    --test_input_file=word_per_line_preliminary_A.jsonl \
    --do_lower_case \
    --output_dir=outputs/ \
    --do_predict \
    --evaluate_during_training \
    --train_max_seq_length=128 \
    --eval_max_seq_length=128 \
    --test_max_seq_length=128 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_test_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --num_train_epochs=6 \
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
    --seed=42
2022-04-04 12:09:47 - INFO - root -   eval_f1_micro_all_entity = 0.8076
2022-04-04 12:09:47 - INFO - root -   eval_f1_micro_without_label_entity = 0.8989
2022-04-04 12:09:47 - INFO - root -   eval_loss = 0.011