# Eddeep

```bash
eddeep_dir=<path-to-eddeep>
data_train_dir=<path-to-train-data-dir>
data_val_dir=<path-to-val-data-dir>
out_dir=<path-to-output-dir>
```

```bash
python ${eddeep_dir}/scripts/train_eddeep_trans.py -t ${data_train_dir}\
                                                   -v ${data_val_dir}\
                                                   -o ${out_dir}/trans\
                                                   -B 2000 -e 50 -as 0.5 -ai 0.5
```
```bash
python ${eddeep_dir}/scripts/train_eddeep_corr.py -t ${data_train_dir}\
                                                  -v ${data_val_dir}\
                                                  -tr ${out_dir}/trans_gen_best.h5\
                                                  -o ${out_dir}/corr\
                                                  -p 1\
                                                  -e 50 -as 0.5 -ai 0.5
```
