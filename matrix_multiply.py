import argparse

import torch


DEFAULT_DIM = 29_952


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated square matrix multiplication on a CUDA GPU."
    )
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dim <= 0:
        raise ValueError("--dim must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    left = torch.randn((args.dim, args.dim), device=device, dtype=torch.float32)
    right = torch.randn((args.dim, args.dim), device=device, dtype=torch.float32)
    result = torch.empty_like(left)

    torch.mm(left, right, out=result)
    torch.cuda.synchronize()

    for count in range(1, args.iterations + 1):
        torch.mm(left, right, out=result)
        torch.cuda.synchronize()
        print(f"success {count} times!", flush=True)


if __name__ == "__main__":
    main()
