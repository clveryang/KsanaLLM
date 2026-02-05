# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import argparse
import csv
import ast
import os
from typing import List, Tuple
from transformers import AutoTokenizer, AutoProcessor, LlamaTokenizer, PreTrainedTokenizerFast
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from benchmark_throughput import PROMPT_AFFIX_DICT


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from the given path"""
    tokenizer_config = get_tokenizer_config(tokenizer_path)
    if tokenizer_config.get("tokenizer_class", "") == "LlamaTokenizer":
        return LlamaTokenizer.from_pretrained(tokenizer_path)
    if tokenizer_config.get("processor_class", "") == "Llama4Processor" \
            and tokenizer_config.get("tokenizer_class", "") == "PreTrainedTokenizer":
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if os.path.exists(tokenizer_path + "/preprocessor_config.json"):
        return AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def build_prompt(tokenizer, prompt: str, model_type: str) -> str:
    """Build prompt with model-specific template"""
    return PROMPT_AFFIX_DICT.get(model_type, "%s").replace("%s", prompt)


def encode_prompt(tokenizer, prompt: str) -> List[int]:
    """Encode prompt to token IDs"""
    return tokenizer.encode(prompt, add_special_tokens=True)


def decode_tokens(tokenizer, tokens: List[int]) -> str:
    """Decode token IDs to text"""
    return tokenizer.decode(tokens, skip_special_tokens=True).rstrip('\ufffd')


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', '--dataset_path',
                        dest='dataset_path',
                        type=str,
                        default="benchmark_input.csv",
                        help='input data for benchmark')
    parser.add_argument('--col_idx',
                        type=int,
                        default=0,
                        help='col_idx to be read from the input csv')
    parser.add_argument('--cache_stat_csv',
                        type=str,
                        default=None,
                        help='cache hit status csv file path')
    parser.add_argument('--output_csv',
                        type=str,
                        default=None,
                        help='visualization result csv file path')
    parser.add_argument('--prompt_num',
                        type=int,
                        default=0,
                        help='Number of input prompts or '
                        'prompt index (count from 0) in single_prompt_mode.')
    parser.add_argument('--single_prompt_mode',
                        action="store_true",
                        help="If on, only process one prompt with index of prompt_num.")
    parser.add_argument('--model_type',
                        type=str,
                        default="llama",
                        choices=[
                            'llama', 'llama-3', 'baichuan', 'qwen', 'vicuna', 'yi',
                            'chatglm', 'empty', 'deepseek_v2', 'deepseek_v3', 'deepseek_r1',
                            'hunyuan_large', 'kimi_k2'
                        ],
                        help="serving model type, used to add prefixes and suffixes"
                             " to the prompt.")
    parser.add_argument('--tokenizer_path',
                        type=str,
                        default=None,
                        help="Path to the tokenizer.")
    args = parser.parse_args()

    return args


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    csv_reader = csv.reader(open(csv_file))
    if remove_head:
        next(csv_reader)
    return [row[col_idx] for row in csv_reader]


def parse_cache_stat(cache_stat_str: str) -> List[Tuple[int, int]]:
    """Parse cache_stat string to List[Tuple[int, int]]"""
    if cache_stat_str == "[]":
        return []
    try:
        return ast.literal_eval(cache_stat_str)
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Failed to parse cache_stat '{cache_stat_str}': {e}")
        return []


# ANSI color codes for terminal output - 方案B：商务色系
class Colors:
    # 柔和背景色 + 黑色字体
    BLACK_TEXT = '\033[30m'  # Black text for better contrast
    PREFIX_BG = '\033[48;5;152m'  # Light gray-green background for prefix cache
    FLEXIBLE_BG_1 = '\033[48;5;153m'  # Light gray-blue background for flexible cache
    FLEXIBLE_BG_2 = '\033[48;5;183m'  # Light gray-purple background for flexible cache
    FLEXIBLE_BG_3 = '\033[48;5;230m'  # Light beige background for flexible cache
    RESET = '\033[0m'  # Reset color

    @staticmethod
    def get_flexible_color(index: int) -> str:
        """Get flexible cache color by index (cycling through 3 colors)"""
        colors = [Colors.FLEXIBLE_BG_1, Colors.FLEXIBLE_BG_2, Colors.FLEXIBLE_BG_3]
        return colors[index % 3]

    @staticmethod
    def print_legend():
        """Print color legend for cache visualization"""
        print("\n" + "="*60)
        print("缓存可视化颜色图例 (Cache Visualization Legend):")
        print("="*60)
        print(f"{Colors.PREFIX_BG}{Colors.BLACK_TEXT} Prefix Cache (前缀缓存) {Colors.RESET} - 浅灰绿色")
        print(f"{Colors.FLEXIBLE_BG_1}{Colors.BLACK_TEXT} Flexible Cache 1 (灵活缓存1) {Colors.RESET} - 浅灰蓝色")
        print(f"{Colors.FLEXIBLE_BG_2}{Colors.BLACK_TEXT} Flexible Cache 2 (灵活缓存2) {Colors.RESET} - 浅灰紫色")
        print(f"{Colors.FLEXIBLE_BG_3}{Colors.BLACK_TEXT} Flexible Cache 3 (灵活缓存3) {Colors.RESET} - 浅米色")
        print("No Cache (无缓存) - 默认颜色")
        print("="*60)


def visualize_cache_hit(input_tokens: List[int], cache_stat: List[Tuple[int, int]],
                        tokenizer) -> str:
    """
    Visualize cache hit status for a single request.

    Args:
        input_tokens: List of input token ids
        cache_stat: List of tuples representing cache hit ranges
        tokenizer: Tokenizer for decoding tokens

    Returns:
        Colored text string showing cache hit status
    """
    if not cache_stat:
        # No cache hit, return plain text
        return decode_tokens(tokenizer, input_tokens)

    total_len = len(input_tokens)
    colored_text = ""
    current_pos = 0

    # Parse cache_stat
    prefix_len = 0
    flexible_ranges = []

    if len(cache_stat) > 0:
        # First tuple is prefix cache
        prefix_len = cache_stat[0][0]  # Both values should be the same
        if len(cache_stat) > 1:
            # Remaining tuples are flexible cache ranges
            flexible_ranges = cache_stat[1:]

    # Build segments: [(start, end, cache_type), ...]
    # cache_type: 0=no cache, 1=prefix, 2+=flexible (index indicates which flexible)
    segments = []

    # Add prefix cache segment
    if prefix_len > 0:
        segments.append((0, prefix_len, 1))  # prefix cache
        current_pos = prefix_len

    # Add flexible cache segments
    flexible_idx = 0
    for start, end in flexible_ranges:
        # Add non-cached segment before this flexible range
        if current_pos < start:
            segments.append((current_pos, start, 0))  # no cache
        # Add flexible cache segment
        segments.append((start, end, 2 + flexible_idx))  # flexible cache
        current_pos = end
        flexible_idx += 1

    # Add remaining non-cached segment
    if current_pos < total_len:
        segments.append((current_pos, total_len, 0))

    # Generate colored output
    for start, end, cache_type in segments:
        if start >= end:
            continue
        segment_tokens = input_tokens[start:end]
        segment_text = decode_tokens(tokenizer, segment_tokens)

        if cache_type == 0:
            # No cache - plain text
            colored_text += segment_text
        elif cache_type == 1:
            # Prefix cache - light gray-green background with black text
            colored_text += Colors.PREFIX_BG + Colors.BLACK_TEXT + segment_text + Colors.RESET
        else:
            # Flexible cache - cycling colors with black text
            flex_color_idx = cache_type - 2
            colored_text += Colors.get_flexible_color(flex_color_idx) + Colors.BLACK_TEXT + segment_text + Colors.RESET

    return colored_text


def save_to_csv(output_csv: str, results: List[Tuple[int, int, str]]):
    """
    Save visualization results to CSV file.

    Args:
        output_csv: Output CSV file path
        results: List of (req_id, input_token_num, colored_text) tuples
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['req_id', 'input_token_num', 'colored_text'])
        for req_id, input_token_num, colored_text in results:
            writer.writerow([req_id, input_token_num, colored_text])


def calculate_cache_stats(cache_stat: List[Tuple[int, int]], total_tokens: int) -> dict:
    """
    Calculate cache hit statistics.
    
    Args:
        cache_stat: List of cache hit ranges
        total_tokens: Total number of input tokens
        
    Returns:
        Dictionary with cache statistics
    """
    if not cache_stat or total_tokens == 0:
        return {
            'prefix_tokens': 0,
            'flexible_tokens': 0,
            'no_cache_tokens': total_tokens,
            'prefix_ratio': 0.0,
            'flexible_ratio': 0.0,
            'total_hit_ratio': 0.0
        }

    prefix_tokens = cache_stat[0][0] if len(cache_stat) > 0 else 0
    flexible_tokens = sum(end - start for start, end in cache_stat[1:]) if len(cache_stat) > 1 else 0
    cached_tokens = prefix_tokens + flexible_tokens
    no_cache_tokens = total_tokens - cached_tokens

    return {
        'prefix_tokens': prefix_tokens,
        'flexible_tokens': flexible_tokens,
        'no_cache_tokens': max(0, no_cache_tokens),
        'prefix_ratio': prefix_tokens / total_tokens * 100,
        'flexible_ratio': flexible_tokens / total_tokens * 100,
        'total_hit_ratio': cached_tokens / total_tokens * 100
    }


def print_cache_summary(stats_list: List[dict]):
    """Print overall cache hit summary"""
    if not stats_list:
        return

    total_requests = len(stats_list)
    avg_prefix_ratio = sum(s['prefix_ratio'] for s in stats_list) / total_requests
    avg_flexible_ratio = sum(s['flexible_ratio'] for s in stats_list) / total_requests
    avg_total_hit_ratio = sum(s['total_hit_ratio'] for s in stats_list) / total_requests

    print("\n" + "="*60)
    print("缓存命中统计摘要 (Cache Hit Summary):")
    print("="*60)
    print(f"总请求数 (Total Requests): {total_requests}")
    print(f"平均前缀缓存命中率 (Avg Prefix Hit Rate): {avg_prefix_ratio:.1f}%")
    print(f"平均灵活缓存命中率 (Avg Flexible Hit Rate): {avg_flexible_ratio:.1f}%")
    print(f"平均总缓存命中率 (Avg Total Hit Rate): {avg_total_hit_ratio:.1f}%")
    print("="*60)


def main(args: argparse.Namespace):
    # Print color legend
    Colors.print_legend()

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Read prompts and cache stats
    prompts = read_from_csv(args.dataset_path, args.col_idx)
    cache_stats_raw = read_from_csv(args.cache_stat_csv)

    # Adjust prompt list length
    prompt_num = min(args.prompt_num, len(cache_stats_raw))
    if args.single_prompt_mode:
        # args.prompt_num counts from 0
        prompt_num = min(args.prompt_num, len(cache_stats_raw) - 1)
        prompts = [prompts[prompt_num]]
        cache_stats_raw = [cache_stats_raw[prompt_num]]
    elif prompt_num > 0:
        prompt_num = min(args.prompt_num, len(cache_stats_raw))
        prompts = prompts[:prompt_num]
        cache_stats_raw = cache_stats_raw[:prompt_num]

    # Ensure the number of prompts matches cache stats
    num_requests = min(len(prompts), len(cache_stats_raw))

    results = []
    cache_stats_summary = []

    for req_id in range(num_requests):
        # Get prompt and cache stat
        prompt = prompts[req_id]
        cache_stat_str = cache_stats_raw[req_id]

        # Build prompt with model-specific template
        built_prompt = build_prompt(tokenizer, prompt, args.model_type)

        # Encode prompt to tokens
        input_tokens = encode_prompt(tokenizer, built_prompt)
        input_token_num = len(input_tokens)

        # Parse cache stat
        cache_stat = parse_cache_stat(cache_stat_str)

        # Calculate cache statistics
        stats = calculate_cache_stats(cache_stat, input_token_num)
        cache_stats_summary.append(stats)

        # Visualize cache hit
        colored_text = visualize_cache_hit(input_tokens, cache_stat, tokenizer)

        # Print result with statistics
        print(f"\nreq_id: {req_id} | tokens: {input_token_num} | "
              f"prefix: {stats['prefix_tokens']}({stats['prefix_ratio']:.1f}%) | "
              f"flexible: {stats['flexible_tokens']}({stats['flexible_ratio']:.1f}%) | "
              f"total_hit: {stats['total_hit_ratio']:.1f}%")
        print("input_text:")
        print(colored_text)

        # Store result for CSV output
        results.append((req_id, input_token_num, colored_text))

    # Print overall summary
    print_cache_summary(cache_stats_summary)

    # Save to CSV if output path is provided
    if args.output_csv is not None:
        save_to_csv(args.output_csv, results)
        print(f"\nVisualization results saved to {args.output_csv}")


if __name__ == "__main__":
    args = args_config()
    main(args)
