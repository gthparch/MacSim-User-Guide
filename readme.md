# Macsim (Revived)
## Introduction
This documentation explains how you can run various GPU workloads on Macsim. Macsim is a trace based cycle-level GPGPU simulator developed by [HPArch](https://sites.gatech.edu/hparch/).

Please feel free to ask questions or point out wrong information to me.

Author: Euijun Chung (echung67@gatech.edu)



## Table of Contents
- [Macsim Installation & Usage](#macsim-installation--usage)
- [Scripts for running Macsim](#scripts-for-running-macsim)
- [List of available traces](#list-of-available-traces)
- [How to create your own trace](#how-to-create-your-own-trace)
- [Known Bugs](#known-bugs)

## Macsim Installation & Usage
### Installation
Installing Macsim is as easy as it gets. Just run the following commands:
```
$ git clone https://github.com/gthparch/macsim.git
$ cd macsim
$ git switch nvbit
$ ./build.py --ramulator -j32
```

If you want to save time and can access `rover`, just copy it from my directory. 
```
$ cd dst/to/your/directory
$ cp /fast_data/echung67/macsim/bin/macsim .
$ cp /fast_data/echung67/macsim/bin/params.in .
$ cp /fast_data/echung67/macsim/bin/trace_file_list .
```
Note that you need three input files: 
- `macsim` (binary executable),
- `params.in` (GPU configuration), and
- `trace_file_list` (list of paths to GPU traces)
to run the Macsim simulation. All these three files should be in the same directory to run Macsim.

### Running
#### 1. Setup the Trace Path

Open trace_file_list and leave 1 on the first line. Change the second line to the path of the trace that you want to run. 

An example of `trace_file_list` would look like this:
```
1
/fast_data/echung67/trace/nvbit/backprop/1024/kernel_config.txt
```
which is for running `backprop` benchmark of `Rodinia Suite` with `1024` configuration.

#### 2. Setup the GPU configuration

Open `params.in` file and put in some numbers for the GPU configuration. An example is at `/fast_data/echung67/macsim/bin/params.in`, and I used this configuration for BNPL paper's evaluation. 

#### 3. Run!

Enter the following command and the simulation results will appear in the current directory.
```
$ ./macsim
```

For example, you can check the total number of cycles in `general.stat.out` file. 


## Scripts for running Macsim
`macsim.py` is a python script that I used to run multiple macsim simulations at the same time.
### Usage
usage: `python3 macsim.py`
```
options:
  -h, --help            show this help message and exit
  --macsim MACSIM       Path to macsim executable
  --params PARAMS       Path to params.in
  --result RESULT       Path to result directory (Default: ./run/)
  --overwrite           Overwrite the simulation results in the result directory
  --suite (benchmark name) 
                        Name of the benchmark suite (valid options: ['vector', 'rodinia', 'tango', 'fastertransformer'])
```

#### Example
The following example will run `vector` and `rodinia` benchmark suite on macsim and save its results to `./run/` directory. 

```
python3 macsim.py --macsim="macsim/bin/macsim" --params="macsim/bin/params.in" --suite vector rodinia --overwrite
```

See `macsim_result.txt` in the result directory for STDOUT and STDERR outputs during the simulation.

> ❗️ You should not change the name of the parameter file `params.in`. The macsim binary will try to find `params.in` file in the same directory and use it as the GPU configuration.

## List of available traces

This is the list of traces that you can access in `rover` machine. I will keep this updated. 
**Suggested Configuration** of each benchmark is the configuration that I used in the BNPL paper.

| Benchmark suite   | Benchmark            | Working on Macsim? | Trace Path | Suggested Config | Source Code |
|-------------------|----------------------|-------------------|-|-|-|
| Vector            | vectoradd            | O                 | /fast_data/echung67/trace/nvbit/vectoradd | 65536 | /fast_data/echung67/trace/source |
|                   | vectormultadd        | O                 | /fast_data/echung67/trace/nvbit/vectormultadd | 65536 | /fast_data/echung67/trace/source |
| Rodinia           | backprop             | O                 | /fast_data/echung67/trace/nvbit/backprop | 524288 | /fast_data/echung67/gpu-rodinia/cuda |
|                   | bfs                  | O                 | /fast_data/echung67/trace/nvbit/bfs | graph256k |
|                   | dwt2d                | O                 | /fast_data/echung67/trace/nvbit/dwt2d | 1024 |
|                   | euler3d              | X                 | X | X |
|                   | gaussian             | O                 | /fast_data/echung67/trace/nvbit/gaussian | matrix128 |
|                   | heartwall            | X                 | X | X |
|                   | hotspot              | O                 | /fast_data/echung67/trace/nvbit/hotspot | r512h2i2 |
|                   | lavaMD               | O                 | /fast_data/echung67/trace/nvbit/lavaMD | 10 |
|                   | lud_cuda             | O                 | /fast_data/echung67/trace/nvbit/lud_cuda | 64 |
|                   | needle               | O                 | /fast_data/echung67/trace/nvbit/needle | 64 |
|                   | nn                   | O                 | /fast_data/echung67/trace/nvbit/nn | 8192k |
|                   | particlefilter_float | O                 | /fast_data/echung67/trace/nvbit/particlefilter_float | 10 |
|                   | particlefilter_naive | O                 | /fast_data/echung67/trace/nvbit/particlefilter_naive | 1000 |
|                   | pathfinder           | O                 | /fast_data/echung67/trace/nvbit/pathfinder | 100 |
|                   | sc_gpu               | O                 | /fast_data/echung67/trace/nvbit/sc_gpu | 10-20-16-64-16-100 |
|                   | srad_v1              | O                 | /fast_data/echung67/trace/nvbit/srad_v1 | 10 |
|                   | srad_v2              | O                 | /fast_data/echung67/trace/nvbit/srad_v2 | 10 |
| Tango             | AlexNet              | O                 | /fast_data/echung67/trace_tango/nvbit/AlexNet | default | /fast_data/echung67/Tango/GPU |
|                   | CifarNet             | O                 | /fast_data/echung67/trace_tango/nvbit/CifarNet | default |
|                   | GRU                  | O                 | /fast_data/echung67/trace_tango/nvbit/GRU | default |
|                   | LSTM                 | O                 | /fast_data/echung67/trace_tango/nvbit/LSTM | default |
|                   | ResNet               | X                 | X | X |
|                   | SqueezeNet           | X                 | X | X |
| FasterTransformer | bert                 | O                 | /data/echung67/trace/nvbit/bert_example | 20 | /fast_data/echung67/FasterTransformer/examples/cpp/ |
|                   | decoding             | O                 | /data/echung67/trace/nvbit/decoding_example | 20 |
|                   | vit                  | X                 | X | X |
|                   | wenet_encoder        | O                 | /data/echung67/trace/nvbit/wenet_encoder_example | 20 |
|                   | xlnet                | X                 | X | X |
| Deepbench         | GEMM                 | O                 | /fast_data/echung67/trace_deep/nvbit/gemm | default | /fast_data/echung67/DeepBench/code/nvidia_small/gemm_bench.cu |
|                   | CNN Inference        | O                 | /fast_data/echung67/trace_deep/nvbit/cnn_inf | default | /fast_data/echung67/DeepBench/code/nvidia_small/conv_bench.cu |
| Pytorch           | Resnet Training      | O                 | /fast_data/echung67/trace_pytorch/nvbit/resnet_train | default | /fast_data/echung67/trace_pytorch/source/resnet_train.py |
|                   | CNN Training         | O                 | /fast_data/echung67/trace_pytorch/nvbit/cnn_train | default | /fast_data/echung67/trace_pytorch/source/cnn_train.py |
|                   | CNN Inference        | O                 | /fast_data/echung67/trace_pytorch/nvbit/cnn_inf | default | /fast_data/echung67/trace_pytorch/source/cnn_inference.py |
|                   | Bert-tiny            | O                 | /fast_data/echung67/trace_pytorch/nvbit/bert_tiny | default | /fast_data/echung67/sandbox/bert/bert_tiny.py |
|                   | Bert-mini            | O                 | /fast_data/echung67/trace_pytorch/nvbit/bert_mini | default | /fast_data/echung67/sandbox/bert/bert_mini.py |
|                   | Bert-small           | O                 | /fast_data/echung67/trace_pytorch/nvbit/bert_small | default | /fast_data/echung67/sandbox/bert/bert_small.py |
|                   | Bert-medium          | O                 | /fast_data/echung67/trace_pytorch/nvbit/bert_medium | default | /fast_data/echung67/sandbox/bert/bert_medium.py |


### Upcoming Plans for trace generations..
| Benchmark suite   | Benchmark            | Working on Macsim? | Trace Path | Suggested Config |
|-------------------|----------------------|--------------------| - | - |
| FasterTransformer | swin                 | ?                  | /data/echung67/trace/nvbit/swin_example | 20 |
|                   | wenet_decoder        | ?                  | /data/echung67/trace/nvbit/wenet_decoder_example | 20 |
| Deepbench         | CNN Training         | -                  | - | - |
|                   | RNN Inference        | -                  | - | - |
|                   | RNN Training         | -                  | - | - |
|                   | Sparse Inference     | -                  | - | - |
|                   | Sparse Training      | -                  | - | - |
| Pytorch           | Resnet Inference     | ?                  | /fast_data/echung67/trace_pytorch/nvbit/resnet_inf | - |
| GraphBig          | -                    | -                  | - | - |
| Gunrock           | -                    | -                  | - | - |
| Tango             | More Block Sizes?    | -                  | - | - |


## How to create your own trace
> ❗️Warning❗️ The trace generation tool for macsim is very unstable, so use at your own risk.

### Installation
> If you are working on rover, you can skip this step.

Please refer to this repository: https://github.com/ejchung0406/gpu-trace-generate

### Usage
This tool works with any GPU programs including CUDA binaries and Tensorflow/Pytorch libraries.
However, you should carefully choose the workload because even a very small workload can be too big for the tool to generate traces.
For instance, training a very small CNN with a few iterations may fill hundreds of GBs and eventually blow up the storage of `rover`.
This will impact other `rover` users so please use this tool wisely.

To generate traces, simply add a few lines to your original command. For instance, if you want to run `python3 cnn_train.py`, the following command will inject trace-generating instructions to the original workload and start generating Macsim traces.
`/fast_data/echung67/nvbit_release/tools/main/main.so` is the path to the trace-generating tool that I wrote, so you can use it freely when you are using `rover`.
```
CUDA_INJECTION64_PATH=/fast_data/echung67/nvbit_release/tools/main/main.so python3 cnn_train.py
```
There are a few arguments that you can use:
- `INSTR_BEGIN`: Beginning of the instruction interval on each kernel where to apply instrumentation. (`default = 0`)
- `INSTR_END`: End of the instruction interval on each kernel where to apply instrumentation. (`default = UINT32_MAX`)
- `KERNEL_BEGIN`: Beginning of the kernel interval where to generate traces. (`default = 0`)
- `KERNEL_END`: End of the kernel interval where to generate traces. (`default = UINT32_MAX`)
- `TOOL_VERBOSE`: Enable verbosity inside the tool. (`default = 0`)
- `TRACE_PATH`: Path to trace file. (`default = './'`)
- `COMPRESSOR_PATH`: Path to the compressor binary file. (`default = '/fast_data/echung67/nvbit_release/tools/main/compress'`)
- `DEBUG_TRACE`: Generate human-readable debug traces together when this value is 1. (`default = 0`)
- `OVERWRITE`: Overwrite the previously generated traces in TRACE_PATH directory when this value is 1. (`default = 0`)

#### Example
```
$ CUDA_INJECTION64_PATH=/fast_data/echung67/nvbit_release/tools/main/main.so \
  TRACE_PATH=./ \
  KERNEL_END=5 \
  DEBUG_TRACE=1 \
  OVERWRITE=1 \
  python3 m.py
```
This command will generate traces for the first 5 CUDA kernels of the workload `python3 m.py`. Also, the tool will overwrite the previous traces and generate the debug traces as well. 

## Known Bugs
> `src/memory.cc:1043: ASSERT FAILED (I=19  C=13193):  0`

**When?** FasterTransformer trace + too many number of cores (40+ cores)

**Solution?** Reduce the number of cores

> `src/factory_class.cc:77: ASSERT FAILED (I=0  C=0):  m_func_table.find(policy) != m_func_table.end()`

**When?** `params.in` file is missing (using wrong file name for `params.in`)

**Solution?** Don't use custom names for GPU config file, use `params.in`.

> `src/process_manager.cc:826: ASSERT FAILED (I=0  C=0):  error opening trace file: ...`

**When?** When too many trace files are open at the same time

**Solution?** Add `ulimit -n 16384` to your `~/.bashrc`.
- However, for traces that use more than 16384 files opened at the same time, it seems like there's no solution for this.. Any helps to solve this issue would be helpful!
