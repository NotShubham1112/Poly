"""
models.polychain.pretraining
Self-supervised pretraining tasks for PolyChain.

Available tasks:
    asterisk_mask : predict the type of masked '*' connection points
    sub_smiles_mask : BERT-style atom masking on the SMILES tokens
"""
from .asterisk_mask import AsteriskMaskHead, asterisk_mask_loss
from .sub_smiles_mask import SubSmilesMaskHead, sub_smiles_mask_loss

__all__ = [
    "AsteriskMaskHead", "asterisk_mask_loss",
    "SubSmilesMaskHead", "sub_smiles_mask_loss",
]
