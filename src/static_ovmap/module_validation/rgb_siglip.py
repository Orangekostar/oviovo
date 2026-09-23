"""Explicit RGB HWC input contract for new S/G/Q image inference."""

from functools import partial

from .region_evidence import FrozenSiglipBackend as _FrozenSiglipBackend


class FrozenSiglipBackend(_FrozenSiglipBackend):
    """Keep frozen model processing while disambiguating narrow RGB crops.

    The historical backend also binds immutable native text receipts. Image
    adapters use this subclass so its correction has a separate source identity.
    """

    def __init__(self, *, model, processor, tokenizer, device):
        super().__init__(
            model=model,
            processor=partial(processor, input_data_format="channels_last"),
            tokenizer=tokenizer,
            device=device,
        )
