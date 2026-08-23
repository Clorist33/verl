import re
import argparse
import os
import numpy as np
from typing import List, Tuple, Union

def extract_metrics_from_log(log_file_path: str) -> Tuple[List[float], List[List[float]], int]:
    """
    提取单个日志文件的有效指标，过滤 Running < 0 的数据
    返回：有效均值列表、所有位置的 acceptance rate 列表（二维）、全量匹配行数
    """
    # 匹配 Mean acceptance length 和 Per-position acceptance rate 及其后的所有数字
    metrics_pattern = re.compile(
        r'Mean acceptance length: (\d+\.\d+|\d+).*?Per-position acceptance rate:\s*([\d.,\s]+?)(?:, Avg Draft|$)'
    )
    running_pattern = re.compile(r'Running: (\d+) reqs')

    mean_accept_list = []
    all_per_pos_values = []   # 每个元素是一个列表，存放该行的所有位置值
    total_matched = 0

    print(f"log_file_path=={log_file_path}")
    # 兼容多种编码读取日志：UTF-8 -> GBK -> UTF-16
    # 注意：utf-8/gbk 必须用 strict，否则遇到 UTF-16 字节流不会抛异常，
    # 会误把乱码当成成功读取，导致后续正则匹配失败。
    lines = None
    for enc in ('utf-8', 'gbk', 'utf-16'):
        try:
            errors = 'replace' if enc == 'utf-16' else 'strict'
            with open(log_file_path, 'r', encoding=enc, errors=errors) as f:
                lines = f.readlines()
            break
        except Exception:
            continue
    if lines is None:
        return [], [], 0

    for line_num in range(1, len(lines)):
        curr_line = lines[line_num].strip()
        prev_line = lines[line_num - 1].strip()

        if "SpecDecoding metrics" not in curr_line:
            continue
        total_matched += 1

        # 检查上一行的 Running 值
        run_match = running_pattern.search(prev_line)
        if not run_match:
            continue
        try:
            running_val = int(run_match.group(1))
        except ValueError:
            continue
        if running_val < 0:
            continue

        # 提取指标
        metric_match = metrics_pattern.search(curr_line)
        if not metric_match:
            continue

        try:
            # Mean acceptance length
            mean_val = float(metric_match.group(1))

            # Per-position acceptance rate 所有数字
            pp_str = metric_match.group(2)   # 例如 "0.540, 0.299, 0.160"
            pp_values = [float(x.strip()) for x in pp_str.split(',') if x.strip()]

            mean_accept_list.append(mean_val)
            all_per_pos_values.append(pp_values)
        except ValueError:
            continue

    return mean_accept_list, all_per_pos_values, total_matched


def calc_single_stat(data_list: List[float]) -> dict:
    """计算单个指标的统计值：均值、最大值、标准差、CDF10%/90%分位数"""
    if not data_list:
        return {
            "mean": "无有效数据",
            "max": "无有效数据",
            "std": "无有效数据",
            "cdf10": "无有效数据",
            "cdf90": "无有效数据",
            "valid_count": 0
        }
    return {
        "mean": round(np.mean(data_list), 4),
        "max": round(np.max(data_list), 4),
        "std": round(np.std(data_list), 4),
        "cdf10": round(np.percentile(data_list, 10), 4),
        "cdf90": round(np.percentile(data_list, 90), 4),
        "valid_count": len(data_list)
    }


def print_single_file_stats(filename: str, mean_list: List[float],
                            all_pp_vals: List[List[float]], total_matched: int) -> str:
    """
    打印单个文件的细节统计信息，并返回 markdown 文本。
    现在对 Per-position acceptance rate 的每个位置分别统计。
    """
    content = []
    content.append(f"## 📄 文件：{filename}")
    content.append(f"全量匹配指标行数：{total_matched}")

    # Mean acceptance length 统计
    mean_stat = calc_single_stat(mean_list)
    content.append("### Mean acceptance length 统计")
    content.append(f"- 有效数据点数：{mean_stat['valid_count']}")
    content.append(f"- 均值：{mean_stat['mean']}")
    content.append(f"- 最大值：{mean_stat['max']}")
    content.append(f"- 标准差：{mean_stat['std']}")
    content.append(f"- CDF 10%分位数（90%数据大于该值）：{mean_stat['cdf10']}")
    content.append(f"- CDF 90%分位数（10%数据大于该值）：{mean_stat['cdf90']}")

    # Per-position acceptance rate 统计（每个位置）
    content.append("### Per-position acceptance rate 统计")
    if not all_pp_vals:
        content.append("无有效数据")
    else:
        # 确定需要处理的位置数（取所有行中的最小长度）
        lengths = [len(vals) for vals in all_pp_vals]
        num_positions = min(lengths)
        if len(set(lengths)) > 1:
            content.append(f"⚠️ 警告：各行的位置数量不一致，仅对前 {num_positions} 个位置统计")

        for pos_idx in range(num_positions):
            pos_list = [vals[pos_idx] for vals in all_pp_vals]
            pos_stat = calc_single_stat(pos_list)
            content.append(f"#### 位置 {pos_idx + 1}")
            content.append(f"- 有效数据点数：{pos_stat['valid_count']}")
            content.append(f"- 均值：{pos_stat['mean']}")
            content.append(f"- 最大值：{pos_stat['max']}")
            content.append(f"- 标准差：{pos_stat['std']}")
            content.append(f"- CDF 10%分位数（90%数据大于该值）：{pos_stat['cdf10']}")
            content.append(f"- CDF 90%分位数（10%数据大于该值）：{pos_stat['cdf90']}")

    content.append("---\n")

    # 终端打印
    for line in content:
        print(line)

    return "\n".join(content)


def generate_markdown_table(log_files: list, folder_path: str) -> str:
    """
    生成简化汇总 Markdown 表格，仅包含：
    - Mean acceptance length 的均值
    - 每个位置的 Per-position acceptance rate 均值
    """
    files_data = []      # 存储每个文件的结果
    max_positions = 0    # 不同文件中最大的位置数

    for filename in log_files:
        file_path = os.path.join(folder_path, filename)
        mean_list, all_pp_vals, _ = extract_metrics_from_log(file_path)

        mean_mean = round(np.mean(mean_list), 4) if mean_list else "N/A"

        # 计算每个位置的均值
        position_means = []
        if all_pp_vals:
            lengths = [len(vals) for vals in all_pp_vals]
            num_pos = min(lengths)  # 取最小长度，保证数据有效
            for i in range(num_pos):
                pos_vals = [vals[i] for vals in all_pp_vals]
                position_means.append(round(np.mean(pos_vals), 4))
        else:
            num_pos = 0

        files_data.append({
            "filename": filename,
            "mean_accept_mean": mean_mean,
            "position_means": position_means,
            "num_positions": num_pos
        })

        if num_pos > max_positions:
            max_positions = num_pos

    # 构建表头
    header = "| 文件名 | Mean acceptance length 均值 |"
    for i in range(1, max_positions + 1):
        header += f" PP pos{i} 均值 |"
    header = header.rstrip('|')
    separator = "| ------ | ------------------------- |"
    for _ in range(max_positions):
        separator += " -------- |"
    separator = separator.rstrip('|')

    rows = [header, separator]

    for data in files_data:
        row = f"| {data['filename']} | {data['mean_accept_mean']} |"
        # 填充每个位置均值
        for i in range(max_positions):
            if i < data['num_positions']:
                row += f" {data['position_means'][i]} |"
            else:
                row += " N/A |"
        row = row.rstrip('|')
        rows.append(row)

    return "\n".join(rows)


def process_folder_and_save_md(folder_path: str):
    """遍历文件夹，处理所有日志文件，打印细节+汇总表，并保存为md文件"""
    log_suffixes = ('.log', '.txt', '.out', '.err')
    log_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.lower().endswith(log_suffixes):
            log_files.append(file)
    log_files.sort()

    if not log_files:
        print("❌ 文件夹内未找到有效日志文件")
        return

    folder_name = os.path.basename(os.path.normpath(folder_path))
    md_filename = f"{folder_name}_日志统计汇总.md"
    md_file_path = os.path.join(folder_path, md_filename)

    md_content = []
    md_content.append(f"# {folder_name} 文件夹日志指标统计")
    md_content.append(f"> 过滤规则：仅保留目标行上一行 Running ≥ 0 的有效数据\n")
    md_content.append("## 一、各文件详细统计\n")

    print(f"📂 开始处理文件夹：{folder_path}")
    print(f"🔍 共找到 {len(log_files)} 个日志文件，开始逐文件分析...\n")

    for filename in log_files:
        file_path = os.path.join(folder_path, filename)
        mean_list, all_pp_vals, total_matched = extract_metrics_from_log(file_path)
        file_detail = print_single_file_stats(filename, mean_list, all_pp_vals, total_matched)
        md_content.append(file_detail)

    print("\n" + "=" * 80)
    print("## 二、全文件汇总Markdown表格（可直接复制使用）")
    print("=" * 80 + "\n")
    summary_table = generate_markdown_table(log_files, folder_path)
    print(summary_table)

    md_content.append("\n## 二、全文件汇总表格\n")
    md_content.append(summary_table)
    md_content.append(f"\n\n> 生成时间：脚本运行自动生成 | 过滤规则：Running ≥ 0")

    try:
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_content))
        print(f"\n✅ 所有统计内容已保存至：{md_file_path}")
    except Exception as e:
        print(f"\n❌ 保存md文件失败：{str(e)}")


def main():
    parser = argparse.ArgumentParser(description='批量日志分析：保留细节打印+汇总表（仅均值），自动保存md文件')
    parser.add_argument('folder_path', type=str, help='日志文件夹绝对/相对路径')
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"❌ 错误：路径 {args.folder_path} 不是有效文件夹")
        return

    process_folder_and_save_md(args.folder_path)


if __name__ == "__main__":
    main()