from pathlib import Path

path = Path('src/massive_scatter/sparse_dataset.py')
text = path.read_text()
text = text.replace('from typing import TypedDict', 'from typing import Any, TypedDict')
text = text.replace('    ) -> dict[str, object]:\n        """Finalize one selected frontier level', '    ) -> dict[str, Any]:\n        """Finalize one selected frontier level')
path.write_text(text)
