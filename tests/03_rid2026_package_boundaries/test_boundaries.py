import ast
from pathlib import Path


def test_rid2026_does_not_import_downstream_packages():
    package_root = Path(__file__).parents[2] / "src" / "rid2026"
    forbidden_packages = ("refdiff", "amc", "rfstyle")
    violations = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name == package or name.startswith(f"{package}.")
                for name in names
                for package in forbidden_packages
            ):
                violations.append(str(path.relative_to(package_root)))
    assert not violations, f"RID2026 imports downstream packages: {violations}"
