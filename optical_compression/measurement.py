"""Shared, actual processor measurements and label-free local-VLM readback."""
import difflib
import json
from pathlib import Path
import unicodedata

from workflow import digest
from .build import SYSTEM, USER
from .document import fingerprint, plain_text


def input_messages(*, image=None, text=None):
    if (image is None) == (text is None):
        raise ValueError('provide exactly one image or text input')
    content = ([{'type': 'image', 'image': str(image)}] if image is not None
               else [{'type': 'text', 'text': text}])
    return [dict(role='system', content=[dict(type='text', text=SYSTEM)]),
            dict(role='user', content=content + [dict(type='text', text=USER)])]


def processor_fingerprint(directory):
    files = {p.name: digest(p) for p in sorted(Path(directory).iterdir())
             if p.is_file() and p.suffix in {'.json', '.txt', '.model', '.jinja', '.py'}}
    if not files:
        raise ValueError('model directory has no processor/tokenizer files')
    return files


class TokenMeter:
    def __init__(self, directory):
        from transformers import AutoProcessor
        self.directory = Path(directory).resolve()
        self.processor = AutoProcessor.from_pretrained(str(self.directory), local_files_only=True)
        self.identity = dict(processor_files=processor_fingerprint(directory),
            protocol_sha256=fingerprint(dict(system=SYSTEM, user=USER,
                text_format='USER PROFILE / CANDIDATE ITEM; canonical normalized fields v1',
                add_generation_prompt=True)))

    def encode(self, messages):
        from ranking.conversation import process_conversation
        return process_conversation(self.processor, messages, add_generation_prompt=True)

    def count(self, messages):
        import torch
        from ranking.conversation import token_counts
        with torch.inference_mode():
            encoded = self.encode(messages)
        result = token_counts(self.processor, encoded)
        if 'image_grid_thw' in encoded:
            result['image_grid_thw'] = encoded['image_grid_thw'].tolist()
        return result

    def text(self, doc):
        return self.count(input_messages(text=plain_text(doc)))

    def image(self, path):
        return self.count(input_messages(image=path))


READ_PROMPT = (
    'Transcribe ALL text from the upper USER PROFILE section and the lower CANDIDATE ITEM '
    'section of this image. Exclude only these two section headings. Preserve all words, '
    'numbers and punctuation in reading order. Do not summarize, infer, or judge relevance. '
    'Return only a JSON object with exactly two string values: {"user":"...","item":"..."}.'
)


def normalized_reading(value):
    return ''.join(unicodedata.normalize('NFKC', value).split())


def reading_result(raw, sections):
    """Report transcription similarity as a diagnostic, without an acceptance threshold."""
    try:
        content = raw.strip()
        if content.startswith('```'):
            content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        result = json.loads(content)
        if (not isinstance(result, dict) or set(result) != {'user', 'item'}
                or not all(isinstance(v, str) for v in result.values())):
            raise ValueError('invalid transcription schema')
    except (ValueError, TypeError, IndexError) as exc:
        return dict(parsed=False, error=str(exc), raw=raw)
    scores = {}
    for name in ['user', 'item']:
        expected = normalized_reading(' '.join(sections[name]))
        observed = normalized_reading(result[name])
        scores[name] = difflib.SequenceMatcher(None, expected, observed, autojunk=False).ratio()
    return dict(parsed=True, section_similarity=scores,
                mean_section_similarity=sum(scores.values()) / len(scores), transcription=result)


class LocalReader:
    """Use the unadapted local base model; no ranking labels or source answers are sent."""
    def __init__(self, meter, device='cuda:0', max_new_tokens=4096):
        self.meter, self.device, self.max_new_tokens = meter, device, max_new_tokens
        self.model = None

    def read(self, path):
        import torch
        if self.model is None:
            from transformers import Qwen3VLForConditionalGeneration
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('local readability check requires a CUDA device or device=cpu')
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.meter.directory), local_files_only=True,
                dtype=torch.bfloat16 if self.device.startswith('cuda') else torch.float32,
                attn_implementation='sdpa', device_map={'': self.device})
            self.model.eval()
        messages = [dict(role='user', content=[dict(type='image', image=str(path)),
                                              dict(type='text', text=READ_PROMPT)])]
        encoded = {k: v.to(self.model.device) for k, v in self.meter.encode(messages).items()}
        with torch.inference_mode():
            generated = self.model.generate(**encoded, do_sample=False,
                max_new_tokens=self.max_new_tokens)
        output = generated[:, encoded['input_ids'].shape[1]:]
        return self.meter.processor.batch_decode(output, skip_special_tokens=True)[0]

    def identity(self):
        """Computed only at export; hashing large weight files is intentionally deferred."""
        files = {p.name: digest(p) for p in sorted(self.meter.directory.glob('*.safetensors'))}
        if not files:
            raise ValueError('local reader model has no safetensors weights')
        return dict(model_weights_sha256=files, prompt_sha256=fingerprint(READ_PROMPT),
                    device=self.device, max_new_tokens=self.max_new_tokens, do_sample=False)
