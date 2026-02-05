# Benchmarking Tool

This tool provides comprehensive benchmarking capabilities for various LLM serving backends including KsanaLLM, vLLM, TensorRT-LLM, and others.

## Features

- **Performance Benchmarking**: Measure throughput, latency, TTFT (Time To First Token), and other metrics
- **Multiple Backends Support**: Compatible with KsanaLLM, vLLM, TensorRT-LLM, SGLang, and more
- **Automatic Diff Checking**: Compare outputs between different benchmark runs to detect consistency issues

# Test Set Description
 - ShareGPT: The [data file](./share_gpt_500.csv) is pre-placed in the current directory. It contains 500 records randomly sampled (using random seed = 0) from the original [ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered). 
    - To use this dataset for benchmarking, simply specify `--dataset-name=sharegpt500`.
    - There is **no need** to explicitly provide `--dataset_path` or `--input_csv`.
 - LongBench V2: The data file should be downloaded manually from [HuggingFace](https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/2b48e49/data.json) before benchmarking. It contains 503 challenging multiple-choice questions with context lengths ranging from 8k to 2M words. 
    - To use this dataset for benchmarking, you need to specify the path to the data file using `--dataset_path`.
    - The dataset supports two prompt settings: Specify `--dataset-name=longbenchV2withCtx` to **include the full background context** in each prompt; Specify `--dataset-name=longbenchV2noCtx` to exclude the context from prompts.
    - When starting the inference server, try to increase `--max-model-len` (if using vLLM) or `max_token_len` (if using KsanaLLM)

# Download model
```
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python

# download huggingface model for example:
# Note: Make sure git-lfs is installed.
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf

```

# Ksana
## Start server
```
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python

export CUDA_VISIBLE_DEVICES=xx

# launch server
python serving_server.py \
    --config_file ../../../examples/ksana_llm2-7b.yaml \
    --port 8080
```
Change config file when trying other options

## Start benchmark
```
cd ${GIT_PROJECT_REPO_ROOT}/benchmarks

# benchmark
python benchmark_throughput.py --input_csv benchmark_input.csv --port 8080 \
    --backend ksana --model_type llama \
    --perf_csv ksana_perf.csv > ksana_stdout.txt 2>&1

# benchmark triton_backend with grpc streaming
python benchmark_throughput.py --host localhost \
    --port 8080 \
    --input_csv benchmark_input.csv  \
    --perf_csv ksana_perf.csv \
    --backend triton-grpc \
    --triton_model_name ksana_llm \
    --tokenizer_path /model_path/ \
    --stream

# benchmark with automatic diff checking between runs
python benchmark_throughput.py --input_csv benchmark_input.csv --port 8080 \
    --backend ksana --model_type llama \
    --perf_csv ksana_perf.csv \
    --enable_diff_check \
    --repeat_num_iters 2 \
    --diff_rouge_threshold 0.5 \
    --diff_mismatch_threshold 10 \
    --diff_output_file comparison_results.txt
```

### Diff Check Parameters
- `--enable_diff_check`: Enable automatic diff checking functionality to compare outputs between two benchmark runs
- `--diff_rouge_threshold`: ROUGE-W threshold below which detailed results are printed (default: 0.5)
- `--diff_mismatch_threshold`: First mismatch position threshold below which detailed results are printed (optional)
- `--diff_output_file`: Output file path for diff results (default: comparison_results.txt)

**Note**: When `--enable_diff_check` is enabled and `--repeat_num_iters` is less than 2, the system will automatically set it to 2 to ensure sufficient runs for comparison.

### Cache Status Analasis Parameters
- `--cache_stat`: Enable cache status analysis and print relevant information
- `--cache_stat_csv`: Output file path for diff results (optional)

**Note**: When `--cache_stat_csv` is given, `--cache_stat` will be automatically set to True.

## Automatic Diff Checking

The diff checking functionality compares output consistency of the same model across multiple runs, helping detect model inference stability and reproducibility issues.

### How It Works

1. **Automatic Run Comparison**: When `--enable_diff_check` is enabled, the tool automatically runs at least two benchmark iterations
2. **Multi-dimensional Comparison**: Uses multiple metrics to evaluate text similarity:
   - **ROUGE-W F1 Score**: Word-level similarity assessment (0-1 range, higher is better)
   - **Levenshtein Distance**: Character-level edit distance (0-1 range, higher is better)
   - **First Mismatch Position**: Detects where text starts to differ (lower indicates earlier divergence)

3. **Smart Filtering**: Only text pairs meeting any of the following conditions are output in detail:
   - ROUGE-W score below the set threshold
   - First mismatch position within the set threshold range

### Use Cases

- **Model Stability Testing**: Verify if the model produces consistent outputs for the same inputs
- **Configuration Change Validation**: Compare model output differences under different configurations
- **Version Regression Testing**: Detect output changes after model updates
- **Randomness Analysis**: Evaluate the degree of randomness in model outputs

### Interpreting Results

The diff check generates a report containing:
- Detailed text comparisons (only showing samples with significant differences)
- Overall statistics (average ROUGE-W score, average edit distance, etc.)
- Trigger condition explanations (which samples were flagged and why)

### Standalone Usage of check_diff.py

You can also use the diff checking tool independently to compare any two CSV files containing text outputs:

```bash
cd ${GIT_PROJECT_REPO_ROOT}/tools/inference_diff_checker

# Basic comparison with default settings
python check_diff.py baseline_results.csv comparison_results.csv

# Show only texts with very low ROUGE-W scores
python check_diff.py baseline_results.csv comparison_results.csv --rouge-threshold 0.2

# Show texts with early mismatches (within first 5 characters)
python check_diff.py baseline_results.csv comparison_results.csv --first-mismatch-threshold 5

# Combine both thresholds and specify custom output file
python check_diff.py baseline_results.csv comparison_results.csv \
    --rouge-threshold 0.4 \
    --first-mismatch-threshold 10 \
    --output detailed_analysis.txt
```

#### CSV File Format

The CSV files should contain text data in the first column:

```csv
"This is the expected output"
"Another reference text"
"Final example text"
```

#### Command Line Parameters

- `csv_file1`: Path to the first CSV file (reference/baseline texts)
- `csv_file2`: Path to the second CSV file (comparison texts)
- `--rouge-threshold`: ROUGE-W threshold below which detailed results are printed (default: 0.5)
- `--first-mismatch-threshold`: First mismatch position threshold below which detailed results are printed (optional)
- `--output` / `-o`: Output file path for results (default: comparison_results.txt)

#### Metrics Explanation

- **ROUGE-W F1 Score**: Higher is better (0-1 range), measures word-level similarity
- **Levenshtein Ratio**: Higher is better (0-1 range), measures character-level similarity
- **First Mismatch Position**: Lower indicates earlier divergence (1-indexed)

Texts are flagged for detailed output if they meet ANY of the threshold conditions.

## Cache Hit Status Visualization

A tool for visualizing LLM inference cache hits with color-coded terminal output. It displays which parts of each prompt hit the prefix cache or flexible cache.

### How It Works

1. **Generate Cache Stats**: Run benchmark with `--cache_stat_csv` to collect cache hit data
2. **Visualize Results**: Use `cache_stat_visualizer.py` to display colored prompts showing cache status
3. **Color Coding**: 
   - Light gray-green = Prefix cache hit
   - Light gray-blue/purple/beige = Flexible cache hits
   - Default color = No cache hit

### Use Cases

- **Cache Performance Analysis**: Understand how effectively your prompts utilize prefix and flexible caching
- **Prompt Engineering**: Optimize prompt structure to maximize cache hits
- **System Tuning**: Validate cache configuration and identify optimization opportunities

### Interpreting Results

Each request displays:
- **Per-request stats**: Token counts and hit rates for prefix cache, flexible cache, and total
- **Colored prompt**: Color-coded text showing which tokens were cached (see legend at start of output)
- **Overall summary**: Average hit rates across all requests

Higher hit rates indicate better cache utilization and potentially faster inference.

### Usage of cache_stat_visualizer.py

```bash
# Step 1: Generate cache stats during benchmark
python benchmark_throughput.py --input_csv benchmark_input.csv --port 8080 \
    --backend ksana --model_type llama \
    --cache_stat_csv cache_stat.csv

# Step 2: Visualize cache hits (copy command from cache_stat.csv header)
python cache_stat_visualizer.py \
    --input_csv benchmark_input.csv \
    --cache_stat_csv cache_stat.csv \
    --model_type llama \
    --tokenizer_path /path/to/tokenizer

# Visualize the first 5 prompt for detailed inspection
python cache_stat_visualizer.py \
    --input_csv benchmark_input.csv \
    --cache_stat_csv cache_stat.csv \
    --model_type llama \
    --tokenizer_path /path/to/tokenizer \
    --prompt_num 5

# Visualize single prompt for detailed inspection
python cache_stat_visualizer.py \
    --input_csv benchmark_input.csv \
    --cache_stat_csv cache_stat.csv \
    --model_type llama \
    --tokenizer_path /path/to/tokenizer \
    --single_prompt_mode \
    --prompt_num 0
```

#### CSV File Format

The `cache_stat_csv` file contains:
- **Header row**: Ready-to-use visualization command
- **Data rows**: Cache hit ranges in format `[(prefix_end, prefix_end), (flex_start1, flex_end1), ...]`

Example:
```csv
python cache_stat_visualizer.py --input_csv input.csv --cache_stat_csv cache.csv ...
[(100, 100), (150, 200)]
[(80, 80), (120, 180), (250, 300)]
```

#### Command Line Parameters

- `--input_csv`: Input prompts CSV file (same as used in benchmark)
- `--cache_stat_csv`: Cache statistics CSV file (generated by benchmark)
- `--model_type`: Model type for prompt template (e.g., llama, qwen, deepseek_v3)
- `--tokenizer_path`: Path to model tokenizer (required)
- `--prompt_num`: Number of prompts to process (default: all) or prompt index in single_prompt_mode (default: 0)
- `--single_prompt_mode`: Process only one specific prompt
- `--output_csv`: Save colored output to CSV (optional)

#### Metrics Explanation

- **Prefix Cache**: Consecutive tokens from the beginning that hit cache (e.g., system prompts)
- **Flexible Cache**: Non-consecutive token ranges that hit cache (e.g., repeated context)
- **Hit Rate**: `(prefix_tokens + flexible_tokens) / total_tokens × 100%`
- **Total Hit Rate**: Overall percentage of tokens served from cache

# vLLM
## Start server
```
export MODEL_PATH=${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python/Llama-2-7b-hf
export CUDA_VISIBLE_DEVICES=xx

python -m vllm.entrypoints.api_server \
     --model $MODEL_PATH \
     --tokenizer $MODEL_PATH \
     --trust-remote-code \
     --max-model-len 1536 \
     --pipeline-parallel-size 1 \
     --tensor-parallel-size 1 \
     --gpu-memory-utilization 0.94 \
     --disable-log-requests \
     --port 8080 
```

## Start benchmark
```
python benchmark_throughput.py --port 8080  --input_csv benchmark_input.csv  \
    --model_type llama \
    --tokenizer_path $MODEL_PATH  \
    --backend vllm \
    --perf_csv vllm_perf.csv > vllm_stdout.txt 2>&1
```
