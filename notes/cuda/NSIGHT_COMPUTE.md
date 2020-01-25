# nsight compute

nsight compute

Replaces `nvprof` for SM 7.5 and later

the ubuntu-packaged nsight-compute didn't work to gather these metrics

1. Had to install nsight-compute 2019.5
  - place in `/usr/local/nvidia/NVIDIA-Nsight-Compute-2019.5/`


## Build as normal

```
cmake --build build --target inOneWeekendCuda
./build/inOneWeekendCuda PR >! output/cu_prof_8x4.ppm
```

## updated metrics!


`inst_fp_32,inst_fp_64` ->

`smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum`


```
sudo /usr/local/nvidia/NVIDIA-Nsight-Compute-2019.5/nv-nsight-cu-cli \
--metrics  smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum \
   -f -o cu_prof_8x4 ./build/inOneWeekendCuda PR >! output/cu_prof_8x4.ppm
```

## analyse

```
/usr/local/nvidia/NVIDIA-Nsight-Compute-2019.5/nv-nsight-cu-cli --import cu_prof_8x4.nsight-cuprof-report
```
