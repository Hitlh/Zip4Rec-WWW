"""Per-configuration immutable base/SFT reference and complete SFT validation gate."""
import argparse
import fcntl
import json
import time
from pathlib import Path
import torch
from .common import digest, grouped, now, rank_metrics, read_json, rows, signature, write_json
from .model_io import setup


def cache_key(plan, split, row):
    # Preserve the main-table training/scoring cache namespace exactly.
    return "dense:" + row["sample_id"]



def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();plan=read_json(a.run/'protocol.json');prepared=Path(plan['prepared']);prepared.mkdir(parents=True,exist_ok=True)
    lock=(prepared/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in plan['code_sha256'].items():assert digest(Path(__file__).parent/name)==h,name
    if (prepared/'complete.json').exists():
        saved=read_json(prepared/'complete.json');assert saved['protocol_sha256']==digest(a.run/'protocol.json')
        for split,h in saved['reference_sha256'].items():assert digest(prepared/f'{split}.jsonl')==h
        assert digest(prepared/'base.json')==saved['base_cache_sha256'];return
    adapter=Path(plan['initial_adapter']);assert digest(adapter/'adapter_model.safetensors')==plan['initial_adapter_sha256']
    model,encode,logits,fingerprint=setup(plan['base_model'],adapter)
    cache_path=prepared/'base.json'
    cache=read_json(cache_path) if cache_path.exists() else dict(base_fingerprint=fingerprint,namespace=plan['experiment'],
        SFT_adapter_sha256=plan['initial_adapter_sha256'],records={})
    assert cache['base_fingerprint']==fingerprint and cache['SFT_adapter_sha256']==plan['initial_adapter_sha256']
    started=time.monotonic();counts={};gate=read_json(prepared/'inner_forward_gate.json') if (prepared/'inner_forward_gate.json').exists() else None
    for split in ['valid','train']:
        source=a.run/('inputs/valid.jsonl' if split=='valid' else 'inputs/train_ten.jsonl');records=rows(source)
        assert digest(source)==plan['input_sha256']['valid' if split=='valid' else 'train_ten']
        assert len(grouped(records))==(plan['train_groups'] if split=='train' else plan['validation_natural_hit_users'])
        output=prepared/f'{split}.jsonl';completed=rows(output) if output.exists() else []
        assert [r['sample_id'] for r in completed]==[r['sample_id'] for r in records[:len(completed)]]
        with output.open('a') as stream:
            for index,row in enumerate(records):
                sig=signature(row);key=cache_key(plan,split,row)
                if index<len(completed):
                    assert completed[index]['input_sha256']==sig
                    # Recover a cache write interrupted after score append by reconstructing from the two original logits.
                    if key not in cache['records']:
                        assert 'base_logits' in completed[index]
                        cache['records'][key]={'input_sha256':sig,'base_logits':completed[index]['base_logits'],'prompt_tokens':completed[index]['prompt_tokens']}
                    assert cache['records'][key]['input_sha256']==sig;continue
                encoded=encode(row)
                with torch.no_grad():
                    with model.disable_adapter():base=logits(encoded).float()
                    adapted=logits(encoded).float();delta=adapted-base
                    if gate is None:
                        from .sft_support import verify_verbalizers
                        from transformers import AutoProcessor
                        processor=AutoProcessor.from_pretrained(plan['base_model'],local_files_only=True)
                        ids=torch.tensor(verify_verbalizers(processor.tokenizer)[:2],device=model.device)
                        full=model(**encoded,use_cache=False,output_hidden_states=True,return_dict=True)
                        head=model.get_base_model().get_output_embeddings()
                        comparison=torch.mv(head.weight.index_select(0,ids).float(),full.hidden_states[-1][0,-1].float())
                        error=float((adapted-comparison).abs().max());assert error<1e-5
                        gate={'maximum_logit_error':error,'verified_utc':now()};write_json(prepared/'inner_forward_gate.json',gate)
                        del full,comparison,processor
                assert torch.isfinite(delta).all();tokens=int(encoded['attention_mask'].sum())
                cache['records'][key]={'input_sha256':sig,'base_logits':base.cpu().tolist(),'prompt_tokens':tokens}
                result={k:row[k] for k in ['sample_id','group_id','user_id','candidate_item_id','target_item_id','candidate_score','label']}
                result.update(input_sha256=sig,base_logits=base.cpu().tolist(),delta_logit_A=float(delta[0]),delta_logit_B=float(delta[1]),
                              condition_score=float(delta[0]-delta[1]),prompt_tokens=tokens)
                stream.write(json.dumps(result)+'\n');completed.append(result)
                if (index+1)%100==0:
                    stream.flush();write_json(cache_path,cache)
                    progress=dict(split=split,scored=index+1,rows=len(records),elapsed_seconds=time.monotonic()-started,gpu=fingerprint['inference_runtime']['gpu_uuid'])
                    write_json(prepared/'live_state.json',progress);print(json.dumps(progress),flush=True)
        write_json(cache_path,cache);counts[split]=len(completed)
        write_json(output.with_suffix('.metadata.json'),dict(complete=True,rows=len(completed),data_sha256=digest(source),
            adapter_sha256=plan['initial_adapter_sha256'],output_sha256=digest(output),fingerprint=fingerprint,created_utc=now()))
        if split=='valid' and plan.get('historical_valid_scores'):
            historical=rows(plan['historical_valid_scores']);input_by_id={r['sample_id']:r for r in records}
            assert len(historical)==len(completed) and {r['sample_id'] for r in historical}==set(input_by_id)
            historical=[{**input_by_id[r['sample_id']],**r} for r in historical]
            current_by_id={r['sample_id']:r for r in completed}
            errors=[abs(r['condition_score']-current_by_id[r['sample_id']]['condition_score']) for r in historical]
            checks={}
            for name,alpha in [('raw',None),('main_fused',plan['historical_main_alpha'])]:
                current,users=rank_metrics(completed,alpha);previous,old_users=rank_metrics(historical,alpha)
                old_map={r['group_id']:r for r in old_users}
                checks[name]=dict(current=current,historical=previous,changed_full_rankings=sum(r['ranking']!=old_map[r['group_id']]['ranking'] for r in users),
                                 changed_target_ranks=sum(r['target_rank']!=old_map[r['group_id']]['target_rank'] for r in users))
            report=dict(max_score_error=max(errors),mean_score_error=sum(errors)/len(errors),comparisons=checks,
                        score_tolerance=1e-4,historical_score_sha256=digest(plan['historical_valid_scores']),fresh_reference=True,
                        passed=max(errors)<1e-4 and all(c['changed_full_rankings']==0 for c in checks.values()))
            write_json(prepared/'baseline_reproduction.json',report)
            assert report['passed'],'SFT baseline mismatch; diagnose inputs/runtime before any GRPO training'
    write_json(prepared/'complete.json',dict(completed_utc=now(),protocol_sha256=digest(a.run/'protocol.json'),gpu=fingerprint['inference_runtime']['gpu_uuid'],fingerprint=fingerprint,
        rows=counts,seconds=time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),base_cache_sha256=digest(cache_path),
        reference_sha256={s:digest(prepared/f'{s}.jsonl') for s in counts}))

if __name__=='__main__':main()
