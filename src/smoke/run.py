import asyncio
import random
import time
import os
import argparse
from typing import List
from statistics import mean
import sys
import random

from tabulate import tabulate
import numpy as np
import openai
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import tiktoken


def generate_random_words(count=10, length=10):
    words = []
    for _ in range(count):
        word = "".join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    return words


def generate_long_text_file(text_file: str):
    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Vestibulum vel dolor sit amet lacus ultrices ullamcorper. "
        "Suspendisse nec justo sit amet nulla dapibus sagittis. "
    )
    words = lorem.split()
    repeat_count = (5000 // len(words)) + 1
    full_text = " ".join(words * repeat_count)[: 5000 * 6]
    with open(text_file, "w") as f:
        f.write(full_text)


def generate_even_sizes(n: int, min_words: int, max_words: int) -> List[int]:
    step = (max_words - min_words) / (n - 1)
    return [int(min_words + i * step) for i in range(n)]


async def run_query(
    session_id: int,
    query_id: int,
    text: str,
    stats: List[dict],
    model_name: str,
    openai_client,
    encoding,
    test_rate_limit=False,
    pbar=None,
    stop_event=None,
):
    if stop_event.is_set():
        return
    try:
        start_time = time.time()
        first_token_time = None
        token_count = 0

        stream = await openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"Summarize the following:\n{text}"}],
            stream=True,
            max_tokens=200,
        )

        async for chunk in stream:
            if stop_event.is_set():
                return
            if not first_token_time:
                first_token_time = time.time()
            content = chunk.choices[0].delta.content or ""
            token_count += len(encoding.encode(content))

        end_time = time.time()
        stats.append(
            {
                "session": session_id,
                "query": query_id,
                "ttft": first_token_time - start_time if first_token_time else None,
                "tps": token_count,
                "success": True,
                "total_time": end_time - start_time,
                "code": 200,
            }
        )

    except Exception as e:
        try:
            code = int(e.code)
        except Exception:
            code = -1

        stats.append(
            {
                "session": session_id,
                "query": query_id,
                "error": str(e),
                "success": False,
                "code": code,
            }
        )
        if code == 429 and test_rate_limit:
            print("Found a 429 error. Rate limiting test passed.")
            stop_event.set()
            raise asyncio.CancelledError()

    if pbar:
        pbar.update(1)


def stats_summary(values: List[float], label: str) -> List:
    return (
        [
            label,
            f"{mean(values):.2f}",
            f"{np.percentile(values, 50):.2f}",
            f"{np.percentile(values, 90):.2f}",
        ]
        if values
        else [label, "-", "-", "-"]
    )


async def user_session(
    session_id: int,
    sizes: List[int],
    stats: List[dict],
    queries_per_user: int,
    base_text: str,
    model_name: str,
    same_text: bool,
    same_text_size: int,
    openai_client,
    encoding,
    test_rate_limit,
    pbar,
    stop_event,
):
    for query_id in range(queries_per_user):
        if stop_event.is_set():
            break
        if same_text:
            truncated_text = base_text[:same_text_size]
        else:
            word_count = random.choice(sizes) - 100
            truncated_text = base_text.split()[:word_count]
            truncated_text.extend(generate_random_words(10, 10))
            truncated_text.shuffle()
            truncated_text = " ".join(truncated_text)

        await run_query(
            session_id,
            query_id,
            truncated_text,
            stats,
            model_name,
            openai_client,
            encoding,
            test_rate_limit,
            pbar,
            stop_event,
        )


async def async_main(args):
    stop_event = asyncio.Event()

    if not os.path.exists(args.text_file):
        generate_long_text_file(args.text_file)

    with open(args.text_file, "r") as f:
        base_text = f.read()

    openai_client = openai.AsyncOpenAI(
        api_key=args.api_key, base_url=args.api_base if args.api_base else None
    )

    try:
        encoding = tiktoken.encoding_for_model(args.model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    stats = []
    if args.single_run:
        print("Running a single test query with 100 words...")
        sample_text = " ".join(base_text.split()[:100])
        await run_query(0, 0, sample_text, stats, args.model, openai_client, encoding)
    else:
        total_queries = args.num_users * args.queries_per_user
        sizes = generate_even_sizes(total_queries, args.min_words, args.max_words)

        pbar = tqdm(total=total_queries, desc="Running queries")

        tasks = [
            asyncio.create_task(
                user_session(
                    i,
                    sizes,
                    stats,
                    args.queries_per_user,
                    base_text,
                    args.model,
                    args.same_text,
                    args.same_text_size,
                    openai_client,
                    encoding,
                    args.test_rate_limit,
                    pbar,
                    stop_event,
                )
            )
            for i in range(args.num_users)
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("Cancelled all remaining tasks due to rate limiting.")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        pbar.close()

    if not args.test_rate_limit:
        success = all(entry.get("success", False) for entry in stats)
    else:
        # we want to find at least one 429
        success = any(entry["code"] == 429 for entry in stats)
        if success:
            print("Found a 429 error. Rate limiting test passed.")
            return 0
        else:
            print("No 429 error. Rate limiting test failed.")
            return 1

    ttf_times = [
        s["ttft"] for s in stats if s.get("success") and s.get("ttft") is not None
    ]
    per_query_tps = [
        s["tps"] / s["total_time"]
        for s in stats
        if s.get("success") and s.get("tps") is not None and s.get("total_time")
    ]

    total_tokens = sum(s["tps"] for s in stats if s.get("success"))
    total_duration = sum(s["total_time"] for s in stats if s.get("success"))
    global_tps = total_tokens / total_duration if total_duration > 0 else 0

    total = len(stats)
    successes = sum(1 for s in stats if s.get("success"))
    failures = total - successes
    table = [
        stats_summary(ttf_times, "Time to First Token (s)"),
        stats_summary(per_query_tps, "Tokens/sec (Per Query)"),
    ]

    errors = [s for s in stats if not s.get("success") and s.get("error")]

    print("\n--- SUMMARY REPORT ---")
    print(f"Total Queries: {total}")
    print(f"Successful Queries: {successes}")
    print(f"Failed Queries: {failures}")
    print(tabulate(table, headers=["Metric", "Mean", "P50", "P90"], tablefmt="grid"))

    print(
        f"\nGlobal Throughput: {global_tps:.2f} tokens/sec across {total_duration:.2f} seconds"
    )
    print("SUCCESS" if success else "FAILURE: Some queries failed")

    if errors:
        print("\n--- FIRST ERROR ---")
        print(f"Session {errors[0]['session']} - Query {errors[0]['query']}")
        print(f"Error: {errors[0]['error']}")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Async OpenAI Query Benchmark")
    parser.add_argument(
        "--num-users", type=int, default=20, help="Number of concurrent users"
    )
    parser.add_argument(
        "--queries-per-user", type=int, default=10, help="Number of queries per user"
    )
    parser.add_argument(
        "--same-text",
        action="store_true",
        help="Use the same text for all queries",
        default=False,
    )
    parser.add_argument(
        "--same-text-size",
        type=int,
        default=500,
        help="Same text size",
    )
    parser.add_argument(
        "--test-rate-limit",
        action="store_true",
        default=False,
        help="Test the rate limiter",
    )

    parser.add_argument(
        "--min-words", type=int, default=500, help="Minimum number of words per query"
    )
    parser.add_argument(
        "--max-words", type=int, default=5000, help="Maximum number of words per query"
    )
    parser.add_argument(
        "--text-file", type=str, default="long_text.txt", help="Path to base text file"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Optional custom base URL for the OpenAI API endpoint",
    )

    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run a single test query with 100 words",
    )

    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    args = parser.parse_args()

    # tests fastly rate limiter
    if args.test_rate_limit:
        args.same_text = True
        args.same_text_size = 10
        args.num_users = 250
        args.queries_per_user = 1

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
