"""One-candidate forwards using the same restricted FP32 A/B head as SFT."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'

import torch
from .common import base_fingerprint


def setup(model_path, adapter, trainable=False, seed=6062, gradient_checkpointing=True):
    from .conversation import process_conversation, record_messages
    from .sft_support import verify_verbalizers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
    from peft import PeftModel
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    ids = torch.tensor(verify_verbalizers(processor.tokenizer)[:2], device='cuda:0')
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0},
        local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
    model = base if adapter is None else PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    if adapter is not None:
        for cfg in model.peft_config.values():
            cfg.lora_dropout = 0.
    if trainable:
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        assert len(params) == 504 and sum(p.numel() for _, p in params) == 21823488
        assert all('language_model.layers.' in n and 'lora_' in n and p.dtype == torch.float32 for n, p in params)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model.eval()
    counts = {'candidate_forward_calls': 0, 'language_layer_forward_calls': 0}
    def count_layer(module, inputs):
        counts['language_layer_forward_calls'] += 1
    original_for_hooks = model if adapter is None else model.get_base_model()
    for name, module in original_for_hooks.model.named_modules():
        if name.startswith('language_model.layers.') and name.split('.')[-1].isdigit():
            module.register_forward_pre_hook(count_layer)

    def encode(row):
        encoded = process_conversation(processor, record_messages(row, include_assistant=False), add_generation_prompt=True)
        length = int(encoded['attention_mask'].sum())
        if length > 2048:
            raise ValueError('prompt too long: ' + row['sample_id'])
        assert encoded['input_ids'].shape[1] == length, 'padding changes BF16 rounding'
        return {k: v.to(model.device) for k, v in encoded.items()}

    def logits(encoded):
        counts['candidate_forward_calls'] += 1
        # Calling the inner multimodal model avoids materializing full-vocabulary logits.
        original = model if adapter is None else model.get_base_model()
        hidden = original.model(**encoded, use_cache=False, return_dict=True).last_hidden_state
        position = int(encoded['attention_mask'].sum()) - 1
        head = original.get_output_embeddings()
        value = torch.mv(head.weight.index_select(0, ids).float(), hidden[0, position].float())
        if getattr(head, 'bias', None) is not None:
            value = value + head.bias.index_select(0, ids).float()
        return value

    logits.counts = counts
    return model, encode, logits, base_fingerprint(model_path)
