"""Validated field presentation: rules can change labels/separators, never values."""
import re


def separator(value):
    if not isinstance(value, str) or not re.fullmatch(r'[ ,;:/|•·()\[\]—–\-]{1,16}', value):
        raise ValueError('separators must be 1..16 visible punctuation/space characters')
    return value


def prepare_document(document, presentation=None):
    """Reconstruct displayed fields from source, checking complete field provenance.

    A rule groups source ids; user text remains verbatim. Structured item values
    retain their original order within each field, while labels are presentation.
    The renderer repeats this operation using its untouched source documents.
    """
    if presentation is None:
        return document
    if not isinstance(presentation, dict) or set(presentation) - {'user', 'item'}:
        raise ValueError('presentation can only configure user and item sections')
    result = {}
    for section in ['user', 'item']:
        fields = document[section]
        if section not in presentation:
            result[section] = fields
            continue
        rules = presentation[section]
        if not isinstance(rules, list) or not rules or len(rules) > len(fields):
            raise ValueError('presentation requires nonempty field groups')
        source = {field['id']: field for field in fields}
        used, output = [], []
        def group_text(rule, depth=0):
            if depth > 4:
                raise ValueError('presentation nesting exceeds four levels')
            if isinstance(rule, dict) and 'groups' in rule:
                groups = rule['groups']
                if (set(rule) - {'groups', 'separator'} or not isinstance(groups, list)
                        or not groups or len(groups) > len(fields)):
                    raise ValueError('invalid nested presentation groups')
                join = separator(rule.get('separator', ' '))
                return join.join(group_text(child, depth + 1) for child in groups)
            allowed = {'ids', 'separator', 'joins'}
            if section == 'item':
                allowed |= {'label', 'list_separator', 'empty_value', 'skip_empty'}
            if not isinstance(rule, dict) or set(rule) - allowed or 'ids' not in rule:
                raise ValueError('unsupported presentation rule keys')
            ids = rule['ids']
            if (not isinstance(ids, list) or not ids or len(ids) > len(fields)
                    or any(not isinstance(key, str) or key not in source for key in ids)):
                raise ValueError('presentation references an unknown source field')
            if any(key in used for key in ids) or len(ids) != len(set(ids)):
                raise ValueError('presentation must include every source field exactly once')
            used.extend(ids)
            join = separator(rule.get('separator', ' '))
            joins = rule.get('joins', {})
            if not isinstance(joins, dict) or set(joins) - set(ids[1:]):
                raise ValueError('joins may only reference later fields in this group')
            for value in joins.values():
                separator(value)
            if section == 'user':
                parts = [source[key]['text'] for key in ids]
            else:
                if any(not {'label', 'values'} <= set(source[key]) for key in ids):
                    raise ValueError('structured item metadata is required for presentation')
                label = rule.get('label', source[ids[0]]['label'] if len(ids) == 1 else None)
                if (not isinstance(label, str) or not label.strip() or len(label) > 120
                        or any(not char.isprintable() for char in label)):
                    raise ValueError('each item group needs a nonempty visible label')
                list_join = separator(rule.get('list_separator', '; '))
                empty = rule.get('empty_value', 'Unknown')
                if empty not in ('', 'Unknown'):
                    raise ValueError('empty_value must be empty or Unknown')
                skip = rule.get('skip_empty', False)
                if type(skip) is not bool:
                    raise ValueError('skip_empty must be boolean')
                parts = [list_join.join(source[key]['values']) if source[key]['values'] else empty
                         for key in ids]
            text = ''
            for key, part in zip(ids, parts):
                if section == 'item' and skip and not part:
                    continue
                if text:
                    text += joins.get(key, join)
                text += part
            if section == 'item':
                text = label + ': ' + text
            return text

        for index, rule in enumerate(rules):
            output.append(dict(id=f'{section}_g{index}', text=group_text(rule)))
        if len(used) != len(source) or set(used) != set(source):
            raise ValueError('presentation must include every source field exactly once')
        result[section] = output
    return result
