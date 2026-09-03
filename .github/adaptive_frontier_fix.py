from pathlib import Path

path = Path("src/massive_scatter/sparse_dataset.py")
text = path.read_text()
text = text.replace("from typing import TypedDict", "from typing import Any, TypedDict")
text = text.replace(
    '    ) -> dict[str, object]:\n        """Finalize one selected frontier level',
    '    ) -> dict[str, Any]:\n        """Finalize one selected frontier level',
)
path.write_text(text)

path = Path("viewer/src/main.ts")
text = path.read_text()
old = """interface ViewResponse {
  origin: [number, number];
  layers: LayerViewResponse[];
}
"""
new = """interface ViewResponse {
  origin: [number, number];
  layers: LayerViewResponse[];
  primitive_count: number;
}
"""
if old not in text:
    raise SystemExit("ViewResponse interface anchor not found")
path.write_text(text.replace(old, new))
