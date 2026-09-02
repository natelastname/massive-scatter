from pathlib import Path

path = Path('src/massive_scatter/builder.py')
text = path.read_text()
text = text.replace(
    '                "viewer requires the shared figure span to remain exactly representable."\n',
    '                "viewer requires the shared figure span to remain exactly "\n'
    '                "representable."\n',
)
path.write_text(text)
