"""Utility: Query Gemini API with a prompt JSONL file.

Takes any prompt JSONL (naming, corpus, requery) and sends each prompt
to the Gemini API. Saves responses as JSONL with the original fields
plus a "response" field.

Usage:
    python -m tools.query_gemini \
        --input  geminon_curation/output/2025_09/prompts/naming_prompts.jsonl \
        --output geminon_curation/output/2025_09/responses/naming_responses.jsonl \
        --api-keys KEY1,KEY2,KEY3 \
        --model gemini-2.5-flash \
        --max-workers 8 \
        --resume
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai


def make_clients(api_keys, model):
    """Create a list of Gemini clients, one per API key."""
    clients = []
    for key in api_keys:
        client = genai.Client(api_key=key)
        clients.append(client)
    return clients


def query_one(client, model, prompt, temperature, max_tokens, seed=None, max_retries=3):
    """Send a single prompt to Gemini with exponential backoff retry."""
    cfg_kwargs = dict(temperature=temperature, max_output_tokens=max_tokens)
    if seed is not None:
        cfg_kwargs["seed"] = seed
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(**cfg_kwargs),
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt + 1
                time.sleep(wait)
            else:
                raise e


def main():
    parser = argparse.ArgumentParser(description="Query Gemini API with prompt JSONL")
    parser.add_argument("--input", type=str, required=True, help="Input prompt JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output response JSONL file")
    parser.add_argument("--api-keys", type=str, required=True,
                        help="Comma-separated Gemini API keys")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional Gemini API seed for reproducibility")
    parser.add_argument("--resume", action="store_true",
                        help="Skip prompts already in output file")
    args = parser.parse_args()

    api_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    if not api_keys:
        print("Error: No API keys provided")
        return

    # Load prompts
    prompts = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts from {args.input}")

    # Load existing responses if resuming
    completed_idxs = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    completed_idxs.add(r["idx"])
        print(f"Resuming: {len(completed_idxs)} already completed")

    remaining = [p for p in prompts if p["idx"] not in completed_idxs]
    if not remaining:
        print("All prompts already completed!")
        return
    print(f"Processing {len(remaining)} prompts with {len(api_keys)} API key(s), {args.max_workers} workers")

    # Create clients
    clients = make_clients(api_keys, args.model)
    client_idx = [0]
    client_lock = threading.Lock()

    def get_client():
        """Round-robin client selection."""
        with client_lock:
            c = clients[client_idx[0] % len(clients)]
            client_idx[0] += 1
            return c

    # Process
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_done = len(completed_idxs)
    n_total = len(prompts)
    n_errors = 0
    write_lock = threading.Lock()

    def process_one(entry):
        nonlocal n_done, n_errors
        client = get_client()
        try:
            # Per-prompt deterministic seed: base seed + prompt idx (so each prompt is independently reproducible)
            entry_seed = (args.seed + entry["idx"]) if args.seed is not None else None
            response_text = query_one(
                client, args.model, entry["prompt"],
                args.temperature, args.max_tokens, entry_seed, args.max_retries,
            )
            result = dict(entry)
            result["response"] = response_text

            with write_lock:
                with open(args.output, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 10 == 0 or n_done == n_total:
                    print(f"  Progress: {n_done}/{n_total} ({n_errors} errors)")
        except Exception as e:
            with write_lock:
                n_errors += 1
                print(f"  Error on idx {entry['idx']}: {e}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_one, entry) for entry in remaining]
        for future in as_completed(futures):
            future.result()  # Raise any unhandled exceptions

    print(f"\nDone! {n_done}/{n_total} completed, {n_errors} errors")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
