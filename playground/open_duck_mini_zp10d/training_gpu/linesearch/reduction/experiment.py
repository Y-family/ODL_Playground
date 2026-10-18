"""Archived baseline and accepted slice variant for numerical regression checks."""

import ast
from pathlib import Path

from playground.common import mjx_compat as compat


def candidates():
    archived = Path(__file__).with_name('accepted_implementation.py').read_text()
    node = next(n for n in ast.parse(archived).body
                if isinstance(n, ast.FunctionDef) and n.name == '_pyramidal_point')
    source = ast.get_source_segment(archived, node)
    before = 'jp.concatenate((quad[:start], friction, quad[stop:]), axis=0)'
    assert source.count(before) == 1
    versions = {
        'accepted': source,
        'update_slice': source.replace(before, 'jax.lax.dynamic_update_slice(quad, friction, (start, 0))'),
    }
    result = {}
    for name, code in versions.items():
        namespace = dict(vars(compat))
        exec(compile(code, f'<{name}>', 'exec'), namespace)
        result[name] = namespace['_pyramidal_point']
        result[name].__source__ = code
    return result
