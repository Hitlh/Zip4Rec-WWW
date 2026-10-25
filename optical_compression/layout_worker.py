"""Resource-limited worker for the restricted layout Python subset."""
import json
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8_000_000, 8_000_000))
    from .layout_engine import Constraints, FontMetrics, validate_source
    payload = json.load(sys.stdin)
    tree = validate_source(payload['source'])
    builtins = {}
    import builtins as standard
    for name in ['min', 'max', 'len', 'range', 'sorted', 'sum', 'round', 'enumerate',
                 'zip', 'int', 'float', 'bool', 'str', 'list', 'dict', 'tuple', 'abs']:
        builtins[name] = getattr(standard, name)
    scope = {'__builtins__': builtins}
    exec(compile(tree, '<layout>', 'exec'), scope)
    cfg = Constraints(**payload['constraints'])
    metrics = FontMetrics(cfg)
    plans = [scope['layout'](doc, metrics, dict(payload['constraints'])) for doc in payload['documents']]
    json.dump(plans, sys.stdout, allow_nan=False)


if __name__ == '__main__':
    main()
