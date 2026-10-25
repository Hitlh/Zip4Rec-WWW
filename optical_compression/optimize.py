"""Initialize a reusable top/bottom layout with ten real training examples and a visual agent."""
import argparse
import json
from pathlib import Path

from workflow import digest
from .document import select_examples
from .layout_engine import Constraints
from .measurement import TokenMeter, LocalReader
from .optimizer.agent import ChatClient, run_agent
from .optimizer.tools import LayoutSession, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True, help='Module-one train_candidates.jsonl (or ten same-schema train examples)')
    p.add_argument('--model', type=Path, required=True, help='Local Qwen3-VL model and processor')
    p.add_argument('--output', type=Path, required=True, help='New local search directory')
    p.add_argument('--prepare-only', action='store_true', help='Measure baseline without calling the agent API or loading VLM weights')
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    agent_config = config['agent']
    if not isinstance(agent_config.get('model'), str) or not agent_config['model'].strip():
        p.error('config.agent.model must name a model with image input AND function calling')
    turns = config.get('max_turns', 40)
    if type(turns) is not int or not 1 <= turns <= 200:
        p.error('max_turns must be between 1 and 200')
    if a.output.exists():
        raise FileExistsError(a.output)
    examples = select_examples(a.samples, count=10)
    constraints = Constraints(**config.get('constraints', {}))
    client = None
    try:
        if not a.prepare_only:
            client = ChatClient(agent_config)
        meter = TokenMeter(a.model)
        reader_config = dict(config.get('reader', {}))
        threshold = config.get('readability_threshold', .90)
        reader_enabled = reader_config.pop('enabled', threshold is not None)
        reader = LocalReader(meter, **reader_config) if reader_enabled else None
        session = LayoutSession(examples, constraints, meter, reader, a.output,
            max_length=config.get('max_length', 2048),
            readability_threshold=threshold,
            max_candidates=config.get('max_candidates', 20),
            text_token_limits=config.get('text_token_limits'),
            initial_source=Path(config['initial_layout']).read_text() if config.get('initial_layout') else None)
        write_json(session.output / 'run_config.json', dict(config=config,
            input_sha256=digest(a.samples), config_sha256=digest(a.config), base_model=str(a.model.resolve())))
        if a.prepare_only:
            cid = session.revise_layout(session.initial_source, 'initial baseline')['candidate_id']
            session.render_and_measure(cid)
            print('Prepared actual pure-text/image measurements and images: ' + str(session.output))
            return
        frozen = run_agent(session, client, model=agent_config['model'], max_turns=turns)
        if frozen is None:
            raise SystemExit('No qualified strategy found. Search records retained; no frozen strategy exported.')
        print('Qualified frozen strategy: ' + str(frozen))
    finally:
        if client is not None:
            client.close()


if __name__ == '__main__':
    main()
