import torch

from src.static_ovmap.attention_runtime import omit_unused_attention_weights, chunk_decoder_queries


def test_unused_attention_weights_preserve_masked_decoder_output():
    torch.manual_seed(0)
    module = torch.nn.MultiheadAttention(32, 4).eval()
    q, k = torch.randn(7, 1, 32), torch.randn(23, 1, 32)
    mask = torch.zeros((7, 23), dtype=torch.bool); mask[:, 15:] = True
    with torch.inference_mode():
        expected = module(q, k, k, attn_mask=mask)[0]
        assert omit_unused_attention_weights(module) == 1
        result, weights = module(q, k, k, attn_mask=mask)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    assert weights is None


def test_query_chunking_keeps_all_keys_and_per_query_masks():
    torch.manual_seed(2)
    module = torch.nn.MultiheadAttention(32, 4).eval()
    q, k = torch.randn(7, 1, 32), torch.randn(23, 1, 32)
    mask = torch.zeros((4, 7, 23), dtype=torch.bool)
    mask[:, ::2, 15:] = True
    mask[:, 1::2, :5] = True
    with torch.inference_mode():
        expected = module(q, k, k, attn_mask=mask)[0]
        assert chunk_decoder_queries(module, 2) == 1
        result, weights = module(q, k, k, attn_mask=mask)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    assert weights is None
