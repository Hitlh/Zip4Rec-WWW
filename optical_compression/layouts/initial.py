def layout(document, metrics, constraints):
    width = min(max(640, constraints['min_width']), constraints['max_width'])
    padding = 12
    return {
        'width': width,
        'padding': padding,
        'line_gap': 4,
        'section_gap': 16,
        'user_rows': metrics.flow(document['user'], width - 2 * padding, pack=False),
        'item_rows': metrics.flow(document['item'], width - 2 * padding, pack=False),
    }
