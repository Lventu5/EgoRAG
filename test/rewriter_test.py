import argparse
import logging
from typing import List
from retrieval.rewriter import QueryRewriterLLM

def run_tests(queries: List[str], model_name: str, device: str):
    rewriter = QueryRewriterLLM(model_name=model_name, device=device)

    for q in queries:
        logging.info("=" * 80)
        logging.info(f"QUERY: {q}")

        rewritten = rewriter(q, modality="default")
        print("\n[Rewritten]")
        print(rewritten)

        decomposed = rewriter(q, modality="decompose")
        print("\n[Decomposed JSON]")
        print(decomposed)
        print()


def build_argparser():
    parser = argparse.ArgumentParser(description="Test QueryRewriterLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HF model name or local path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device preference (auto-downgrade to cpu if no CUDA).")
    parser.add_argument("--query", type=str, nargs="*", default=["A man shouts while a car explodes"], help="One or more test queries.")
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
    )
    args = build_argparser().parse_args()

    test_queries = [
        # 1. Speech + explosion
        "A man shouting as a car explodes",

        # 2. Cooking sounds + visual action
        "A woman frying vegetables while music plays in the background",

        # 3. Speech + crowd + visual sports scene
        "A commentator speaking loudly as players score a goal",

        # 4. Environmental noise + movement
        "A train passes while people talk on the platform",

        # 5. Musical performance
        "A band performing live on stage with cheering fans",

        # 6. Domestic context + clear audio cues
        "A baby crying as a vacuum cleaner runs in the room",

        # 7. Nature scene with sound
        "Birds singing as the sun rises over a forest",

        # 8. Action sequence with mixed modalities
        "A man running through rain while thunder strikes",

        # 9. Human conversation + background TV sound
        "Two people arguing while the TV news plays in the background",

        # 10. Emotional scene with ambient sound
        "A woman crying softly as waves crash on the shore"
    ]
    run_tests(test_queries, model_name=args.model, device=args.device)