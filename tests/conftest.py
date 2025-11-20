import warnings

warnings.filterwarnings(
    "ignore",
    message="Cannot initialize NVML",
    module="torch.cuda",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="onnxscript.converter",
)
warnings.filterwarnings(
    "ignore",
    message="# 'dynamic_axes' is not recommended",
    category=UserWarning,
    module="torch.onnx",
)
