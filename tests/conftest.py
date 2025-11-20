import warnings

warnings.filterwarnings(
    "ignore",
    message="Cannot initialize NVML",
    module="torch.cuda",
)
warnings.filterwarnings(
    "ignore",
    message="Expression.__init__ got an unexpected keyword argument 'lineno'",
    category=DeprecationWarning,
    module="onnxscript.converter",
)
warnings.filterwarnings(
    "ignore",
    message="Expression.__init__ got an unexpected keyword argument 'col_offset'",
    category=DeprecationWarning,
    module="onnxscript.converter",
)
warnings.filterwarnings(
    "ignore",
    message="# 'dynamic_axes' is not recommended",
    category=UserWarning,
    module="torch.onnx",
)
