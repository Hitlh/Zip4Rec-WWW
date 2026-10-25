"""Record actual runtime so numerical reference caches stay tied to one environment."""
import importlib.metadata
import subprocess


def runtime_fingerprint():
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    prop = torch.cuda.get_device_properties(0)
    uuid = str(prop.uuid)
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True).stdout.splitlines()[0]
    return dict(gpu_name=prop.name, gpu_uuid=uuid, driver_version=driver,
                compute_capability=f'{prop.major}.{prop.minor}', torch=torch.__version__,
                torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                transformers=importlib.metadata.version('transformers'),
                peft=importlib.metadata.version('peft'), attention='sdpa',
                base_dtype='bfloat16', restricted_head_dtype='float32', AMP=False,
                cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
