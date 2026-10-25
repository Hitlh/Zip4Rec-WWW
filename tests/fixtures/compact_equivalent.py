def layout(document, metrics, constraints):
    width = 640
    compact_prefixes = ['Preferred eras:', 'Moods/styles:', 'Diversity:',
                        'Evidence strength:', 'Uncertainties:']
    user_ids = [field['id'] for field in document['user']]
    joins = {}
    previous_compact = False
    for field in document['user']:
        compact = False
        for prefix in compact_prefixes:
            if field['text'][:len(prefix)] == prefix:
                compact = True
        if compact and previous_compact:
            joins[field['id']] = ' • '
        previous_compact = compact
    by_key = {field['key']: field['id'] for field in document['item']}
    presentation = {
        'user': [{'ids': user_ids, 'separator': ' ', 'joins': joins}],
        'item': [
            {'ids': [by_key['title']], 'label': 'Title'},
            {'groups': [
                {'ids': [by_key['artists']], 'label': 'Artists', 'list_separator': ', '},
                {'ids': [by_key['genres']], 'label': 'Genres', 'list_separator': ', '},
                {'ids': [by_key['era'], by_key['moods']], 'label': 'Era / styles',
                 'list_separator': ', ', 'separator': '; ', 'empty_value': '', 'skip_empty': True},
            ], 'separator': ' • '},
            {'ids': [by_key['summary']], 'label': 'Summary'},
        ],
    }
    fields = metrics.prepare(document, presentation)
    return {
        'width': width,
        'padding': 10,
        'margins': {'left': 10, 'right': 10, 'top': 12, 'bottom': 4},
        'line_height': 16,
        'line_gap': 0,
        'text_anchor': 'la',
        'text_origin': 'font',
        'height_multiple': 32,
        'min_height': 32,
        'trim_rows': True,
        'section_gap': 4,
        'presentation': presentation,
        'user_rows': metrics.flow(fields['user'], 620, pack=False, wrap='words'),
        'item_rows': metrics.flow(fields['item'], 620, pack=False, wrap='words'),
    }
