import asyncio
import aiohttp
import random
import time
import os
import openai
from typing import List
from statistics import mean

# Constants
NUM_CONCURRENT_USERS = 20
QUERIES_PER_USER = 10
MIN_WORDS = 500
MAX_WORDS = 5000
TEXT_FILE = "long_text.txt"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"  # Change as needed

# Ensure OpenAI client is set up
openai_client = openai.AsyncOpenAI(api_key=API_KEY)


# Generate base text file (~5000 words)
def generate_long_text_file():
    if os.path.exists(TEXT_FILE):
        return
    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Vestibulum vel dolor sit amet lacus ultrices ullamcorper. "
        "Suspendisse nec justo sit amet nulla dapibus sagittis. "
    )
    words = lorem.split()
    repeat_count = (5000 // len(words)) + 1
    full_text = " ".join(words * repeat_count)[: 5000 * 6]  # estimate ~6 chars per word
    with open(TEXT_FILE, "w") as f:
        f.write(full_text)


# Load base text
with open(TEXT_FILE, "r") as f:
    base_text = f.read()


# Generate evenly distributed sizes
def generate_even_sizes(n: int, min_words: int, max_words: int) -> List[int]:
    step = (max_words - min_words) / (n - 1)
    sizes = [int(min_words + i * step) for i in range(n)]
    return sizes


# Individual request function
async def run_query(session_id: int, query_id: int, text: str, stats: List[dict]):
    try:
        start_time = time.time()
        first_token_time = None
        token_count = 0

        stream = await openai_client.chat.completions.create(
            model=MODEL_NAME,
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


# Per-user session
async def user_session(session_id: int, sizes: List[int], stats: List[dict]):
    tasks = []
    for query_id in range(QUERIES_PER_USER):
        word_count = random.choice(sizes)
        truncated_text = " ".join(base_text.split()[:word_count])
        tasks.append(run_query(session_id, query_id, truncated_text, stats))
    await asyncio.gather(*tasks)


# Main
async def main():
    generate_long_text_file()
    stats = []
    sizes = generate_even_sizes(
        NUM_CONCURRENT_USERS * QUERIES_PER_USER, MIN_WORDS, MAX_WORDS
    )
    tasks = [user_session(i, sizes, stats) for i in range(NUM_CONCURRENT_USERS)]
    await asyncio.gather(*tasks)

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


if __name__ == "__main__":
    asyncio.run(main())
