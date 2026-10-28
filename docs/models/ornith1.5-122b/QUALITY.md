# Ornith 1.5 122B Quality Verification

## Numerical Parity

- **Dequantization Accuracy:** Bit-identical reference verification against Python ROCmFPX codebook and `charlie12345/ROCmFPX` specification.
- **Dynamic Vocabulary:** Automatic derivation from `token_embd.weight` tensor shape ensuring support for varying tokenizer expansions without hardcoded bounds.
- **Layer Allocation:** 36 GatedDeltaNet SSM recurrent layers + 12 Full Attention layers validated for hidden residual norm stability across long context sequences.
- **Expert Routing:** Top-8 routing verification passes with zero NaN/Inf under FP16 activation accumulation and FP32 residual scaling.
