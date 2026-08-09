from pathlib import Path

import pytest


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    """Provide an isolated directory for image-generation tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
