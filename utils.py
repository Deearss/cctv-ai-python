import torch

def get_device():
    """Ngecek HW trus nentuin device (CUDA/CPU) paling oke."""
    print("--- Pengecekan Hardware ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"MANTAP! GPU Terdeteksi: {gpu_name}")
        return "cuda"
    
    print("GPU Gak Nemu. Lu pake CPU? Lemot ntar!")
    return "cpu"
