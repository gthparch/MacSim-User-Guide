import sys
import os
import argparse
import subprocess
import shutil

VALID_SUITE_NAMES = ["vector", "rodinia", "tango", "fastertransformer", "deepbench", "pytorch"]

def process_options():
  parser = argparse.ArgumentParser(description='macsim.py')
  parser.add_argument('--macsim', required=True, help='Path to macsim executable')
  parser.add_argument('--params', required=True, help='Path to params.in')
  parser.add_argument('--result', default=f"{os.getcwd()}/run", help='Path to result directory')
  parser.add_argument('--overwrite', action='store_true', help='Overwrite the simulation results')
  parser.add_argument('--suite', required=True, nargs='+', type=str, choices=VALID_SUITE_NAMES,
                    help=f"Name of the benchmark suite (valid options: {VALID_SUITE_NAMES})")
  return parser

def run(argv):
  global args

  # parse arguments
  parser = process_options()
  try:
    args = parser.parse_args()
  except argparse.ArgumentError:
    print("Invalid suite name. Valid options are:")
    for suite_name in VALID_SUITE_NAMES:
        print(f"{suite_name} ")
    exit()
  current_dir = os.getcwd()

  ## path to binary
  macsim_files = [args.macsim, args.params]
  result_dir = args.result

  vector_benchmark_names = [
    "vectoradd",
    "vectormultadd",
  ]

  vector_benchmark_subdir = {
    "vectoradd": ["4096", "16384", "65536"],
    "vectormultadd": ["4096", "16384", "65536"],
  }

  rodinia_benchmark_names = [
    # Rodinia
    "backprop",
    "bfs",
    "dwt2d",
    # "euler3d",
    "gaussian",
    # "heartwall",
    # "hotspot",
    "lavaMD",
    "lud_cuda",
    "needle",
    "nn",
    "particlefilter_float",
    "particlefilter_naive",
    "pathfinder",
    "sc_gpu",
    "srad_v1",
    "srad_v2",
  ]

  rodinia_benchmark_subdir = {
    # Rodinia
    "backprop": ["128", "256", "512", "1024", "2048", "4096", "8192", "16384",
                 "32768", "65536", "131072", "262144", "524288", "1048576"],
    "bfs": ["graph1k", "graph2k", "graph4k", "graph8k", "graph16k", "graph32k",
              "graph64k", "graph128k", "graph256k"], # "graph512k"],
    "dwt2d": ["192", "1024"],
    "euler3d": ["fvcorr.domn.097K"],
    "gaussian": ["matrix3", "matrix4", "matrix16", "matrix32", "matrix48", "matrix64", "matrix80", "matrix96", "matrix112", "matrix128"],
    "heartwall": ["frames10"],
    "hotspot": ["r512h512i100", "r512h512i1000", "r512h2i2"],
    "lavaMD": ["1", "2", "3", "5", "7", "10"],
    "lud_cuda": ["64"], # "256", "512"],
    "needle": ["32", "64"], # "128"],
    "nn": ["64k", "128k", "256k", "512k", "1024k", "2048k", "4096k", "8192k"], # "16384k", "32768k"],
    "particlefilter_float": ["10"],
    "particlefilter_naive": ["1000"],
    "pathfinder": ["10", "50", "100"],
    "sc_gpu": ["2-5-4-16-16-32", "3-3-4-16-16-4", "10-20-16-64-16-100"],
    "srad_v1": ["3", "6", "10"],
    "srad_v2": ["10"],  
  }

  tango_benchmark_names = [
    "AlexNet",
    "CifarNet",
    "GRU",
    "LSTM",
  ]

  tango_benchmark_subdir = {
    "AlexNet": ["default"],
    "CifarNet": ["default"],
    "GRU": ["default"],
    "LSTM": ["default"],
  }

  ft_benchmark_names = [
    "bert_example",
    "decoding_example",
    "swin_example",
    "wenet_decoder_example",
    "wenet_encoder_example",
  ]

  ft_benchmark_subdir = {
    "bert_example": ["20"],
    "decoding_example": ["20"],
    "swin_example": ["20"],
    "wenet_decoder_example": ["20"],
    "wenet_encoder_example": ["20"],
  }

  deep_benchmark_names = [
    "gemm",
    # "cnn_inf",
  ]

  deep_benchmark_subdir = {
    "gemm": ["default"],
    "cnn_inf": ["default"],
  }

  torch_benchmark_names = [
    # "cnn_train",
    "cnn_inf",
    "resnet_train",
    # "resnet_inf",
  ]

  torch_benchmark_subdir = {
    "cnn_train": ["default"],
    "cnn_inf": ["default"],
    "resnet_train": ["default"],
    "resnet_inf": ["default"],
  }

  if args.suite:
    for suite_name in args.suite:
      if suite_name == 'vector':
        trace_path_base = "/fast_data/echung67/trace/nvbit/"
        benchmark_names = vector_benchmark_names
        benchmark_subdir = vector_benchmark_subdir
      elif suite_name == 'rodinia':
        trace_path_base = "/fast_data/echung67/trace/nvbit/"
        benchmark_names = rodinia_benchmark_names
        benchmark_subdir = rodinia_benchmark_subdir
      elif suite_name == 'tango':
        trace_path_base = "/fast_data/echung67/trace_tango/nvbit/"
        benchmark_names = tango_benchmark_names
        benchmark_subdir = tango_benchmark_subdir
      elif suite_name == 'fastertransformer':
        trace_path_base = "/data/echung67/trace/nvbit/"
        benchmark_names = ft_benchmark_names
        benchmark_subdir = ft_benchmark_subdir
      elif suite_name == 'deepbench':
        trace_path_base = "/fast_data/echung67/trace_deep/nvbit/"
        # trace_path_base = "/fast_data/echung67/sandbox/trace/deepbench/"
        benchmark_names = deep_benchmark_names
        benchmark_subdir = deep_benchmark_subdir
      elif suite_name == 'pytorch':
        trace_path_base = "/fast_data/echung67/trace_pytorch/nvbit/"
        # trace_path_base = "/fast_data/echung67/sandbox/trace/pytorch/"
        benchmark_names = torch_benchmark_names
        benchmark_subdir = torch_benchmark_subdir
      else:
        exit()

      for bench_name in benchmark_names:
        bench_subdirs = benchmark_subdir[bench_name]
        for bench_subdir in bench_subdirs:
          print(f"[Macsim] Running {suite_name}/{bench_name}/{bench_subdir}")
          # create the result directory
          subdir = os.path.join(result_dir, suite_name, bench_name, bench_subdir)
          if os.path.exists(subdir):
            if args.overwrite:
              shutil.rmtree(subdir)
              os.makedirs(subdir)
            else:
              print(f"{subdir} already exists!")
              continue
          else:
            os.makedirs(subdir)

          trace_path = os.path.join(trace_path_base, bench_name, bench_subdir, "kernel_config.txt")
          if not os.path.exists(trace_path):
            print(f"{trace_path} doesn't exist!")
            continue

          for macsim_file in macsim_files:
            if not os.path.exists(os.path.join(current_dir, macsim_file)):
              print(f"{os.path.join(current_dir, macsim_file)} doesn't exist!")
              return
            os.system(f"cp {os.path.join(current_dir, macsim_file)} {subdir}")
          
          with open(f"{subdir}/trace_file_list", "w") as f:
            f.write(f"1\n" + trace_path)

          cmd = "nohup ./macsim > macsim_result.txt 2>&1"
          print(f"[cmd]: {cmd}")
          subprocess.Popen([f"ulimit -n 16384 && {cmd}"], shell=True, cwd=subdir)
  return

if __name__ == '__main__':
  os.system("ulimit -n 16384")

  run(sys.argv)

  print(" ")

