"""Runtime-only adaptation for released decoders that discard attention weights."""


def omit_unused_attention_weights(net):
    import torch

    def without_weights(module, args, kwargs):
        if module.training or len(args) > 3:
            raise ValueError('eval decoder with keyword attention options required')
        return args, {**kwargs, 'need_weights': False}

    count = 0
    for module in net.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module.register_forward_pre_hook(without_weights, with_kwargs=True)
            count += 1
    return count


def chunk_decoder_queries(net, query_chunk=16):
    """Evaluate independent attention query rows in chunks, retaining every key/value."""
    import torch
    if query_chunk < 1:
        raise ValueError('positive query chunk required')

    def wrap(module):
        original = module.forward

        def forward(query, key, value, **kwargs):
            if module.training or module.batch_first or kwargs.get('is_causal', False):
                raise ValueError('eval sequence-first noncausal decoder attention required')
            options = {**kwargs, 'need_weights': False}
            mask = options.get('attn_mask')
            results = []
            for start in range(0, len(query), query_chunk):
                current = dict(options)
                if mask is not None:
                    current['attn_mask'] = mask[..., start:start+query_chunk, :]
                results.append(original(query[start:start+query_chunk], key, value, **current)[0])
            return torch.cat(results, dim=0), None

        module.forward = forward

    count = 0
    for module in net.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            wrap(module)
            count += 1
    return count
