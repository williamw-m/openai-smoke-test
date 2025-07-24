import asyncio
import random
import time
import os
import argparse
from typing import List
from statistics import mean

import openai
from tqdm.asyncio import tqdm_asyncio  # for async progress bar
from tqdm import tqdm


def generate_long_text_file(text_file: str):
    if os.path.exists(text_file):
        return

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
    pbar=None,
):
    try:
        start_time = time.time()
        first_token_time = None
        token_count = 0

        stream = await openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"Summarize the following:\n{text}"}],
            stream=True,
            max_tokens=500,
        )

        async for chunk in stream:
            if not first_token_time:
                first_token_time = time.time()
            content = chunk.choices[0].delta.content or ""
            token_count += len(content.split())

        end_time = time.time()
        stats.append(
            {
                "session": session_id,
                "query": query_id,
                "ttft": first_token_time - start_time if first_token_time else None,
                "tps": token_count / (end_time - (first_token_time or start_time)),
                "success": True,
            }
        )

    except Exception as e:
        stats.append(
            {
                "session": session_id,
                "query": query_id,
                "error": str(e),
                "success": False,
            }
        )

    if pbar:
        pbar.update(1)


async def user_session(
    session_id: int,
    sizes: List[int],
    stats: List[dict],
    queries_per_user: int,
    base_text: str,
    model_name: str,
    openai_client,
    pbar,
):
    tasks = []
    for query_id in range(queries_per_user):
        word_count = random.choice(sizes)
        truncated_text = " ".join(base_text.split()[:word_count])
        tasks.append(
            run_query(
                session_id,
                query_id,
                truncated_text,
                stats,
                model_name,
                openai_client,
                pbar,
            )
        )
    await asyncio.gather(*tasks)


async def async_main(args):
    generate_long_text_file(args.text_file)

    with open(args.text_file, "r") as f:
        base_text = f.read()

    openai_client = openai.AsyncOpenAI(api_key=args.api_key)
    stats = []
    total_queries = args.num_users * args.queries_per_user
    sizes = generate_even_sizes(total_queries, args.min_words, args.max_words)

    pbar = tqdm(total=total_queries, desc="Running queries")

    tasks = [
        user_session(
            i,
            sizes,
            stats,
            args.queries_per_user,
            base_text,
            args.model,
            openai_client,
            pbar,
        )
        for i in range(args.num_users)
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    success = all(entry.get("success", False) for entry in stats)
    ttf_times = [
        s["ttft"] for s in stats if s.get("success") and s.get("ttft") is not None
    ]
    tps_speeds = [
        s["tps"] for s in stats if s.get("success") and s.get("tps") is not None
    ]

    print("\n--- SUMMARY REPORT ---")
    print(f"Total Queries: {len(stats)}")
    print(f"Successful Queries: {sum(1 for s in stats if s.get('success'))}")
    print(f"Failed Queries: {sum(1 for s in stats if not s.get('success'))}")
    if ttf_times:
        print(f"Average Time to First Token: {mean(ttf_times):.2f} seconds")
    if tps_speeds:
        print(f"Average Tokens Per Second: {mean(tps_speeds):.2f} t/s")
    print("SUCCESS" if success else "FAILURE: Some queries failed")


def main():
    parser = argparse.ArgumentParser(description="Async OpenAI Query Benchmark")
    parser.add_argument(
        "--num-users", type=int, default=20, help="Number of concurrent users"
    )
    parser.add_argument(
        "--queries-per-user", type=int, default=10, help="Number of queries per user"
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
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
