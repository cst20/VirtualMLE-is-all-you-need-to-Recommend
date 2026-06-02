#!/usr/bin/env python3

import argparse
import gzip
import json
import math
import os
from collections import defaultdict
from statistics import mean, median


def detect_gz_files(raw_data_root):
    files = []
    if not os.path.isdir(raw_data_root):
        return files
    for filename in sorted(os.listdir(raw_data_root)):
        if filename.startswith("reviews_") and filename.endswith("_5.json.gz"):
            files.append(os.path.join(raw_data_root, filename))
    return files


def domain_from_gz_path(gz_path):
    filename = os.path.basename(gz_path)
    return filename[len("reviews_") : -len("_5.json.gz")]


def decompress_gz_file(gz_path, extracted_root):
    os.makedirs(extracted_root, exist_ok=True)
    domain = domain_from_gz_path(gz_path)
    output_path = os.path.join(extracted_root, f"{domain}_5.json")
    with gzip.open(gz_path, "rt", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
    return output_path


def load_interactions(json_path):
    user_to_events = defaultdict(list)
    raw_items = set()
    raw_interactions = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            user = obj.get("reviewerID")
            item = obj.get("asin")
            timestamp = obj.get("unixReviewTime")
            if user is None or item is None or timestamp is None:
                continue

            try:
                timestamp = int(timestamp)
            except (TypeError, ValueError):
                continue

            user_to_events[str(user)].append((timestamp, line_no, str(item)))
            raw_items.add(str(item))
            raw_interactions += 1

    return user_to_events, len(raw_items), raw_interactions


def build_sequences(user_to_events, min_len, max_len):
    user_to_seq = {}
    removed_short = 0
    truncated_users = 0

    for user, events in user_to_events.items():
        events.sort(key=lambda x: (x[0], x[1]))
        seq = [item for _, _, item in events]
        if len(seq) < min_len:
            removed_short += 1
            continue
        if len(seq) > max_len:
            seq = seq[-max_len:]
            truncated_users += 1
        user_to_seq[user] = seq

    return user_to_seq, removed_short, truncated_users


def reindex_users_and_items(user_to_seq):
    sorted_users = sorted(user_to_seq.keys())
    all_items = sorted({item for seq in user_to_seq.values() for item in seq})

    user_map = {user: idx for idx, user in enumerate(sorted_users, start=1)}
    item_map = {item: idx for idx, item in enumerate(all_items, start=1)}

    reindexed_sequences = []
    for raw_user in sorted_users:
        new_user = user_map[raw_user]
        new_seq = [item_map[item] for item in user_to_seq[raw_user]]
        reindexed_sequences.append((new_user, new_seq))

    return reindexed_sequences, user_map, item_map


def percentile(sorted_values, p):
    if not sorted_values:
        return 0
    rank = math.ceil((p / 100.0) * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def compute_stats(reindexed_sequences, raw_user_count, raw_item_count, raw_interactions, min_len, max_len, removed_short, truncated_users):
    lengths = sorted(len(seq) for _, seq in reindexed_sequences)
    filtered_interactions = sum(lengths)
    filtered_items = len({item for _, seq in reindexed_sequences for item in seq})

    return {
        "raw_users": raw_user_count,
        "raw_items": raw_item_count,
        "raw_interactions": raw_interactions,
        f"removed_short_users_lt_{min_len}": removed_short,
        f"truncated_users_gt_{max_len}": truncated_users,
        "filtered_users": len(reindexed_sequences),
        "filtered_items": filtered_items,
        "filtered_interactions": filtered_interactions,
        "min_sequence_length": min_len,
        "max_sequence_length": max_len,
        "truncate_strategy": f"keep_last_{max_len}",
        "avg_len": round(mean(lengths), 2) if lengths else 0,
        "median_len": int(median(lengths)) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "p50": percentile(lengths, 50),
        "p75": percentile(lengths, 75),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
    }


def write_sequence_txt(output_root, domain, reindexed_sequences):
    domain_dir = os.path.join(output_root, domain)
    os.makedirs(domain_dir, exist_ok=True)
    sequence_path = os.path.join(domain_dir, "sequential_data_processed.txt")
    with open(sequence_path, "w", encoding="utf-8") as f:
        for user_id, seq in reindexed_sequences:
            f.write(f"{user_id} {' '.join(map(str, seq))}\n")
    return sequence_path


def write_domain_summary(output_root, domain, source_json, sequence_file, stats):
    domain_dir = os.path.join(output_root, domain)
    os.makedirs(domain_dir, exist_ok=True)
    summary_path = os.path.join(domain_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"domain: {domain}\n")
        f.write(f"source_json: {source_json}\n")
        f.write(f"sequence_file: {sequence_file}\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    return summary_path


def write_global_summary(output_root, results):
    summary_path = os.path.join(output_root, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"domain: {result['domain']}\n")
            f.write(f"source_gz: {result['source_gz']}\n")
            f.write(f"extracted_json: {result['extracted_json']}\n")
            f.write(f"sequence_file: {result['sequence_file']}\n")
            f.write(f"domain_summary: {result['domain_summary']}\n")
            for key, value in result['stats'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    return summary_path


def process_domain(gz_path, extracted_root, output_root, min_len, max_len):
    domain = domain_from_gz_path(gz_path)
    extracted_json = decompress_gz_file(gz_path, extracted_root)
    user_to_events, raw_item_count, raw_interactions = load_interactions(extracted_json)
    user_to_seq, removed_short, truncated_users = build_sequences(user_to_events, min_len, max_len)
    reindexed_sequences, user_map, item_map = reindex_users_and_items(user_to_seq)
    sequence_file = write_sequence_txt(output_root, domain, reindexed_sequences)
    stats = compute_stats(
        reindexed_sequences,
        raw_user_count=len(user_to_events),
        raw_item_count=raw_item_count,
        raw_interactions=raw_interactions,
        min_len=min_len,
        max_len=max_len,
        removed_short=removed_short,
        truncated_users=truncated_users,
    )
    domain_summary = write_domain_summary(output_root, domain, extracted_json, sequence_file, stats)

    return {
        "domain": domain,
        "source_gz": gz_path,
        "extracted_json": extracted_json,
        "sequence_file": sequence_file,
        "domain_summary": domain_summary,
        "user_map_size": len(user_map),
        "item_map_size": len(item_map),
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Build sequential domain data from Amazon core-5 raw files")
    parser.add_argument("--raw_data_root", type=str, default="data/raw_data", help="Directory containing reviews_*.json.gz")
    parser.add_argument("--extracted_root", type=str, default="data/raw_data/extracted", help="Directory for decompressed json files")
    parser.add_argument("--output_root", type=str, default="data/sequential_domains", help="Output root for domain txt and summary files")
    parser.add_argument("--min_len", type=int, default=5, help="Minimum sequence length to keep")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length; keep only the last max_len items")
    args = parser.parse_args()

    gz_files = detect_gz_files(args.raw_data_root)
    os.makedirs(args.extracted_root, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    results = []
    for gz_path in gz_files:
        result = process_domain(gz_path, args.extracted_root, args.output_root, args.min_len, args.max_len)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    global_summary = write_global_summary(args.output_root, results)
    print(json.dumps({"processed_domains": [x["domain"] for x in results], "summary": global_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
