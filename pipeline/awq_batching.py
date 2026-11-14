"""Batch token-independent AWQ replay without changing calibration samples.

LLM Compressor 0.15 stores each example separately. Replaying hundreds of tiny
Linear/MoE forwards for each grid point is dominated by Python and CUDA launches.
Only callers that prove the parent is token-independent may use this function.
Attention, masked loss and unknown argument layouts retain upstream execution.
"""
from contextlib import contextmanager

REPLAY_TOKEN_LIMIT = 32768


@contextmanager
def reuse_evaluation_mode(model):
    """Avoid repeated whole-model walks inside an already active eval context.

The caller must keep calibration_forward_context active around the entire
smoothing phase. All Qwen forwards in this phase preserve module training flags.
Non-False calls retain the original train() behavior; restoration is exception-safe.
"""
    if any(module.training for module in model.modules()):
        raise ValueError('AWQ evaluation mode must be established before replay')
    original_train = model.train
    had_local_method = 'train' in vars(model)
    local_method = vars(model).get('train')

    def train(mode=True):
        if mode is False and model.training is False:
            return model
        return original_train(mode)

    model.train = train
    try:
        yield
    finally:
        if had_local_method:
            model.train = local_method
        else:
            del model.train


def batch_tokenwise_cache(cache, token_limit=REPLAY_TOKEN_LIMIT):
    import torch
    from llmcompressor.pipelines.cache import IntermediateValue

    if getattr(cache, '_inferboost_batched', False):
        return False
    batches = cache.batch_intermediates
    if len(batches) < 2:
        return False
    if token_limit <= 0:
        raise ValueError('AWQ replay token limit must be positive')
    key, signature, tensors = None, None, []
    for batch in batches:
        if len(batch) != 1:
            return False
        name, entry = next(iter(batch.items()))
        tensor = entry.value
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
            return False
        if tensor.ndim not in (2, 3) or tensor.numel() == 0:
            return False
        current = (name, tensor.shape[-1], tensor.ndim, tensor.dtype, entry.device)
        if signature is not None and current != signature:
            return False
        key, signature = name, current
        tensors.append(tensor)

    width, rank, device = signature[1], signature[2], signature[4]
    merged, pieces, tokens = [], [], 0

    def flush():
        nonlocal pieces, tokens
        if not pieces:
            return
        tensor = torch.cat(pieces, dim=0)
        if rank == 3:
            tensor = tensor.unsqueeze(0)
        merged.append({key: IntermediateValue(tensor, device)})
        pieces, tokens = [], 0

    for tensor in tensors:
        flat = tensor.reshape(-1, width)
        offset = 0
        while offset < flat.shape[0]:
            count = min(token_limit - tokens, flat.shape[0] - offset)
            pieces.append(flat[offset:offset + count])
            tokens += count
            offset += count
            if tokens == token_limit:
                flush()
    flush()
    # Commit after validation and construction; unsupported layouts stay intact.
    cache.batch_intermediates = merged
    cache._inferboost_batched = True
    if device is not None and torch.device(device).type == 'cuda':
        for index in range(len(cache)):
            cache.pin_memory(index)
    return True
