"""Check the public entry point's local links, assets, and shell examples."""
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = [ROOT / "README.md", ROOT / "docs/QUICKSTART.md"]


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_entry_links_and_anchors_exist(document):
    text = document.read_text()
    links = re.findall(r"\]\(([^)]+)\)", text)
    links += re.findall(r'(?:href|src)="([^"]+)"', text)
    for link in links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            continue
        target = document.parent / unquote(parsed.path) if parsed.path else document
        assert target.is_file(), f"Missing link target: {link} in {document.name}"
        if parsed.fragment:
            headings = re.findall(r"^#+\s+(.+)$", target.read_text(), re.M)
            anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
            assert parsed.fragment in anchors, f"Missing anchor: {link}"


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_gpu_visibility_examples_are_single_shell_values(document):
    for value in re.findall(r"^export CUDA_VISIBLE_DEVICES=(.*)$", document.read_text(), re.M):
        assert value == "0,1,2,3,4,5,6,7"


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_bash_examples_parse(document):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash not installed")
    for command in re.findall(r"```bash\n(.*?)```", document.read_text(), re.S):
        result = subprocess.run([bash, "-n"], input=command, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_paper_and_overview_are_packaged_assets():
    assert (ROOT / "docs/paper.pdf").read_bytes().startswith(b"%PDF-")
    assert (ROOT / "assets/vetta-overview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_readme_is_paper_first_without_author_header():
    text = (ROOT / "README.md").read_text()
    assert text.splitlines()[0] == (
        "# VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents"
    )
    assert re.findall(r"^## (.+)$", text, re.M)[0] == "Overview"
    assert "<h1" not in text and "<p align" not in text
    assert "Jiaju Chen" not in text


def test_four_result_panels_and_composite_are_png_assets():
    names = ["benchmark-comparison", "results-alfworld-15b", "results-webshop-15b",
             "results-alfworld-7b", "results-webshop-7b"]
    for name in names:
        assert (ROOT / "assets" / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
