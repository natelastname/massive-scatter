from pathlib import Path

script = Path('.github/patch_precompute_ranges.py')
text = script.read_text()
needle = '''text = text.replace(
    "getColor: datum => datumColor(datum, data, false),",
    "getColor: datum => datumColor(datum, false, colorRange, alphaRange),",
)
'''
addition = needle + '''text = text.replace(
    "getFillColor: datum => datumColor(datum, data, false),",
    "getFillColor: datum => datumColor(datum, false, colorRange, alphaRange),",
)
'''
if needle not in text:
    raise SystemExit('color accessor patch anchor not found')
script.write_text(text.replace(needle, addition))
exec(compile(script.read_text(), str(script), 'exec'))
