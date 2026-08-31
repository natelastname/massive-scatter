from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

path = Path("viewer/src/main.ts")
text = path.read_text()
text = text.replace(
    "import {aggregateCellCorner} from './lod-cell';\n",
    "import {aggregateCellCorner} from './lod-cell';\n"
    "import {LatestRequestRunner} from './latest-request';\n",
)
text = text.replace(
    "let requestTimer: number | undefined;\n"
    "let activeRequest: AbortController | null = null;\n",
    "let requestTimer: number | undefined;\n"
    "const viewRequests = new LatestRequestRunner<ViewResponse>();\n",
)
start = text.index("async function requestView() {")
end = text.index("\nfunction responseData(", start)
replacement = """function requestView() {
  if (!manifest || plot.clientWidth < 1 || plot.clientHeight < 1) return;
  const bounds = visibleBounds();
  const body = {
    xmin: bounds.minX,
    xmax: bounds.maxX,
    ymin: bounds.minY,
    ymax: bounds.maxY,
    width: plot.clientWidth,
    height: plot.clientHeight,
    max_points: Math.max(1, Number(maxPointsInput.value) || 200_000),
    max_cells: 200_000,
  };
  status.textContent = 'loading viewport…';

  viewRequests.enqueue(
    async () => {
      const response = await fetch('/api/view', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
      return (await response.json()) as ViewResponse;
    },
    response => {
      currentResponse = response;
      renderLayer(response);
    },
    error => {
      status.textContent = `request failed: ${String(error)}`;
    },
  );
}
"""
text = text[:start] + replacement + text[end:]
path.write_text(text)

repo = os.environ["REPO"]
branch = os.environ["BRANCH"]
headers = {
    "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json",
}


def request(method: str, url: str, payload: dict[str, object] | None = None):
    req = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
        method=method,
    )
    with urlopen(req) as response:
        return json.load(response)


relative = path.as_posix()
current = request(
    "GET",
    f"https://api.github.com/repos/{repo}/contents/{quote(relative)}?ref={quote(branch)}",
)
request(
    "PUT",
    f"https://api.github.com/repos/{repo}/contents/{quote(relative)}",
    {
        "message": "Use single-flight viewport requests [skip patch]",
        "content": base64.b64encode(path.read_bytes()).decode(),
        "sha": current["sha"],
        "branch": branch,
    },
)
