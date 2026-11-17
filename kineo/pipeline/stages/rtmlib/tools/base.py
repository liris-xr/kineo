import os
from abc import ABCMeta, abstractmethod
from typing import Any

import cv2
import numpy as np
import torch

from .file import download_checkpoint
def check_mps_support():
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        return 'MPSExecutionProvider' in providers or 'CoreMLExecutionProvider' in providers
    except ImportError:
        return False

RTMLIB_SETTINGS = {
    'opencv': {
        'cpu': (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),

        # You need to manually build OpenCV through cmake
        'cuda': (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
    },
    'onnxruntime': {
        'cpu': 'CPUExecutionProvider',
        'cuda': 'CUDAExecutionProvider',
        'rocm': 'ROCMExecutionProvider',
        'mps': 'CoreMLExecutionProvider' if check_mps_support() else 'CPUExecutionProvider'
    },
}

def build_execution_providers(device: str,
                              use_tensorrt: bool = False,
                              fp16: bool = False,
                              engine_cache_dir: str | None = None) -> list:
    is_cuda = device == 'cuda' or device.startswith('cuda:')

    if fp16 and not use_tensorrt:
        raise ValueError(
            'fp16 requires TensorRT: the CUDA provider runs the model as it '
            'was exported, so the request would be silently ignored.')
    if use_tensorrt and not is_cuda:
        raise ValueError(f'TensorRT requires a CUDA device, got {device}.')

    if not is_cuda:
        return [RTMLIB_SETTINGS['onnxruntime'][device]]

    cuda_provider = 'CUDAExecutionProvider'
    if ':' in device:
        cuda_provider = (cuda_provider,
                         {'device_id': int(device.split(':')[-1])})

    if not use_tensorrt:
        return [cuda_provider]

    tensorrt_options = {
        'trt_fp16_enable': fp16,
        'trt_engine_cache_enable': engine_cache_dir is not None,
    }
    if engine_cache_dir is not None:
        tensorrt_options['trt_engine_cache_path'] = engine_cache_dir

    return [('TensorrtExecutionProvider', tensorrt_options), cuda_provider]


class BaseTool(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 backend: str = 'opencv',
                 device: str = 'cpu',
                 use_tensorrt: bool = False,
                 fp16: bool = False,
                 engine_cache_dir: str | None = None,
                 strict_providers: bool = False):

        if not os.path.exists(onnx_model):
            onnx_model = download_checkpoint(onnx_model)

        if backend == 'opencv':
            try:
                providers = RTMLIB_SETTINGS[backend][device]

                session = cv2.dnn.readNetFromONNX(onnx_model)
                session.setPreferableBackend(providers[0])
                session.setPreferableTarget(providers[1])
                self.session = session
            except Exception:
                raise RuntimeError(
                    'This model is not supported by OpenCV'
                    ' backend, please use `pip install'
                    ' onnxruntime` or `pip install'
                    ' onnxruntime-gpu` to install onnxruntime'
                    ' backend. Then specify `backend=onnxruntime`.')  # noqa

        elif backend == 'onnxruntime':
            import onnxruntime as ort
            if engine_cache_dir is not None:
                os.makedirs(engine_cache_dir, exist_ok=True)
            providers = build_execution_providers(
                device,
                use_tensorrt=use_tensorrt,
                fp16=fp16,
                engine_cache_dir=engine_cache_dir)

            self.session = ort.InferenceSession(path_or_bytes=onnx_model,
                                                providers=providers)

            # onnxruntime silently falls back on provider failure
            if strict_providers:
                requested = providers[0]
                if isinstance(requested, tuple):
                    requested = requested[0]
                resolved = self.session.get_providers()[0]
                if resolved != requested:
                    raise RuntimeError(
                        f'Requested {requested} for {onnx_model} but '
                        f'onnxruntime resolved {resolved}.')

        elif backend == 'openvino':
            from openvino import Core
            core = Core()
            model_onnx = core.read_model(model=onnx_model)

            if device != 'cpu':
                print('OpenVINO only supports CPU backend, automatically'
                      ' switched to CPU backend.')

            self.compiled_model = core.compile_model(
                model=model_onnx,
                device_name='CPU',
                config={'PERFORMANCE_HINT': 'LATENCY'})
            self.input_layer = self.compiled_model.input(0)
            self.output_layer0 = self.compiled_model.output(0)
            self.output_layer1 = self.compiled_model.output(1)

        else:
            raise NotImplementedError

        print(f'load {onnx_model} with {backend} backend')

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # build input to (1, 3, H, W)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        # run model
        if self.backend == 'opencv':
            outNames = self.session.getUnconnectedOutLayersNames()
            self.session.setInput(input)
            outputs = self.session.forward(outNames)
        elif self.backend == 'onnxruntime':
            sess_input = {self.session.get_inputs()[0].name: input}
            sess_output = []
            for out in self.session.get_outputs():
                sess_output.append(out.name)

            outputs = self.session.run(sess_output, sess_input)
        elif self.backend == 'openvino':
            results = self.compiled_model(input)
            output0 = results[self.output_layer0]
            output1 = results[self.output_layer1]
            outputs = [output0, output1]

        return outputs

    def inference_gpu(
            self,
            batch: torch.Tensor,
            output_shapes: list[tuple] | None = None) -> list[torch.Tensor]:
        """Run the model on a contiguous float32 CUDA tensor via IO binding."""
        if self.backend != 'onnxruntime':
            raise RuntimeError(
                f'Device inference needs the onnxruntime backend, got '
                f'{self.backend}.')

        device = batch.device
        device_id = device.index or 0

        binding = self.session.io_binding()
        binding.bind_input(
            name=self.session.get_inputs()[0].name,
            device_type='cuda',
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(batch.shape),
            buffer_ptr=batch.data_ptr())

        outputs = self.session.get_outputs()

        if output_shapes is None:
            for output in outputs:
                binding.bind_output(output.name, 'cuda')
            self.session.run_with_iobinding(binding)
            # No dlpack export in this onnxruntime build
            return [
                torch.as_tensor(value.numpy(), device=device)
                for value in binding.get_outputs()
            ]

        buffers = [
            torch.empty(shape, dtype=torch.float32, device=device)
            for shape in output_shapes
        ]
        for output, buffer in zip(outputs, buffers):
            binding.bind_output(
                name=output.name,
                device_type='cuda',
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(buffer.shape),
                buffer_ptr=buffer.data_ptr())

        self.session.run_with_iobinding(binding)

        return buffers
