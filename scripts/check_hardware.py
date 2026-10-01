import torch


def main() -> None:
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if available:
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
            properties = torch.cuda.get_device_properties(index)
            print(f"GPU {index} memory: {properties.total_memory / 1024**3:.2f} GB")
    else:
        print("No NVIDIA CUDA GPU detected; CPU fallback is active.")


if __name__ == "__main__":
    main()