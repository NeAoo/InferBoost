"""Read live vLLM quantization objects without relying on requested flags."""


def loaded_quantization(module):
    method = getattr(module, 'base_quant_method', None)
    if method is None:
        method = getattr(module, 'quant_method', None)
    # vLLM 0.33 stores compressed-tensors schemes on the layer itself.
    scheme = getattr(module, 'scheme', None)
    if scheme is None and method is not None:
        scheme = getattr(method, 'scheme', None)
    kernel = getattr(method, 'kernel', None) if method is not None else None
    if kernel is None and scheme is not None:
        kernel = getattr(scheme, 'kernel', None)
    return method, scheme, kernel
