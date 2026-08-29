from pathlib import Path

style = Path("viewer/src/style.css")
text = style.read_text()
old = "#plot { position: absolute; inset: 0; }"
new = """#plot {
  position: absolute;
  left: 52px;
  right: 18px;
  top: 18px;
  bottom: 30px;
  overflow: hidden;
}"""
if text.count(old) != 1:
    raise RuntimeError("expected one full-canvas #plot rule")
style.write_text(text.replace(old, new))

main = Path("viewer/src/main.ts")
text = main.read_text()
old = """  const availableWidth = Math.max(1, plot.clientWidth - 100);
  const availableHeight = Math.max(1, plot.clientHeight - 80);
"""
new = """  // The deck canvas already occupies exactly the interior axes rectangle.
  // Keep only a small visual pad so edge markers are not clipped by the frame.
  const availableWidth = Math.max(1, plot.clientWidth - 24);
  const availableHeight = Math.max(1, plot.clientHeight - 24);
"""
if text.count(old) != 1:
    raise RuntimeError("expected one home-fit padding block")
main.write_text(text.replace(old, new))
