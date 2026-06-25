"""Validate transformer docstring Parameters against ``_parameter_constraints``."""

from __future__ import annotations

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "chemotools"

PARAM_HEADER_RE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<spec>.+?)\s*$"
)
SECTION_BOUNDARY_RE = re.compile(r"\n[A-Za-z][A-Za-z0-9_ ]+\n[-]{3,}\s*\n")
QUOTED_TOKEN_RE = re.compile(r"[\"']([A-Za-z0-9_-]+)[\"']")


def _base_names(class_node: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
        elif isinstance(base, ast.Subscript):
            value = base.value
            if isinstance(value, ast.Name):
                names.append(value.id)
            elif isinstance(value, ast.Attribute):
                names.append(value.attr)
    return names


def _is_public_transformer(class_node: ast.ClassDef) -> bool:
    return "TransformerMixin" in _base_names(
        class_node
    ) and not class_node.name.startswith("_")


def _extract_parameters_lines(doc: str) -> list[str]:
    marker = "Parameters\n----------"
    if marker not in doc:
        return []

    block = doc.split(marker, 1)[1]
    boundary = SECTION_BOUNDARY_RE.search(block)
    if boundary is not None:
        block = block[: boundary.start()]
    return block.splitlines()


def _find_header_candidates(lines: list[str]) -> list[tuple[int, str, str]]:
    header_candidates: list[tuple[int, str, str]] = []
    for line in lines:
        match = PARAM_HEADER_RE.match(line)
        if match is None:
            continue
        indent_len = len(match.group("indent"))
        name = match.group("name")
        spec = match.group("spec")
        header_candidates.append((indent_len, name, spec))
    return header_candidates


def _parse_parameters_with_indent(
    lines: list[str], declaration_indent: int
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = PARAM_HEADER_RE.match(line)
        if match is not None and len(match.group("indent")) == declaration_indent:
            if current_name is not None:
                parsed[current_name] = "\n".join(current_lines)
            current_name = match.group("name")
            current_lines = [match.group("spec")]
            continue

        if current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        parsed[current_name] = "\n".join(current_lines)

    return parsed


def _parse_doc_parameters(doc: str) -> dict[str, str]:
    lines = _extract_parameters_lines(doc)
    if not lines:
        return {}

    header_candidates = _find_header_candidates(lines)

    if not header_candidates:
        return {}

    declaration_indent = min(indent for indent, _, _ in header_candidates)
    return _parse_parameters_with_indent(lines, declaration_indent)


def _get_constraints_dict_node(class_node: ast.ClassDef) -> ast.Dict | None:
    for statement in class_node.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "_parameter_constraints"
            and isinstance(statement.value, ast.Dict)
        ):
            return statement.value

        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Dict):
            for target in statement.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_parameter_constraints"
                ):
                    return statement.value

    return None


def _allows_none(value_node: ast.AST) -> bool:
    elements = value_node.elts if isinstance(value_node, ast.List) else [value_node]
    for element in elements:
        if isinstance(element, ast.Constant) and element.value is None:
            return True
    return False


def _extract_str_options(value_node: ast.AST) -> set[str]:
    elements = value_node.elts if isinstance(value_node, ast.List) else [value_node]
    for element in elements:
        if not isinstance(element, ast.Call):
            continue

        function_name: str | None = None
        if isinstance(element.func, ast.Name):
            function_name = element.func.id
        elif isinstance(element.func, ast.Attribute):
            function_name = element.func.attr

        if function_name != "StrOptions" or not element.args:
            continue

        first_arg = element.args[0]
        if not isinstance(first_arg, ast.Set):
            continue

        return {
            item.value
            for item in first_arg.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }

    return set()


def _parse_constraints(constraints_dict: ast.Dict) -> dict[str, ast.AST]:
    parsed: dict[str, ast.AST] = {}
    for key_node, value_node in zip(constraints_dict.keys, constraints_dict.values):
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            parsed[key_node.value] = value_node
    return parsed


def _load_public_transformers() -> list[tuple[Path, ast.ClassDef]]:
    classes: list[tuple[Path, ast.ClassDef]] = []
    for file_path in PACKAGE_ROOT.rglob("*.py"):
        source = file_path.read_text(encoding="utf-8")
        module = ast.parse(source)
        for class_node in module.body:
            if isinstance(class_node, ast.ClassDef) and _is_public_transformer(
                class_node
            ):
                classes.append((file_path, class_node))
    return classes


def test_transformer_docstring_parameter_constraints_are_consistent() -> None:
    """Ensure public transformer docs stay aligned with runtime constraints."""
    mismatches: list[str] = []

    for file_path, class_node in _load_public_transformers():
        class_doc = ast.get_docstring(class_node) or ""
        doc_parameters = _parse_doc_parameters(class_doc)

        constraints_dict = _get_constraints_dict_node(class_node)
        if constraints_dict is None:
            # Some classes may inherit constraints from parent classes.
            continue

        constraints = _parse_constraints(constraints_dict)

        missing_in_docs = sorted(set(constraints) - set(doc_parameters))
        if missing_in_docs:
            mismatches.append(
                f"{file_path.relative_to(PROJECT_ROOT)}::{class_node.name}: "
                f"constraint params missing from docs: {missing_in_docs}"
            )

        extra_in_docs = sorted(set(doc_parameters) - set(constraints))
        if extra_in_docs:
            mismatches.append(
                f"{file_path.relative_to(PROJECT_ROOT)}::{class_node.name}: "
                f"doc params missing from _parameter_constraints: {extra_in_docs}"
            )

        for param_name, value_node in constraints.items():
            if param_name not in doc_parameters:
                continue

            doc_block = doc_parameters[param_name]
            doc_lower = doc_block.lower()

            allowed_choices = _extract_str_options(value_node)
            if allowed_choices:
                documented_tokens = set(QUOTED_TOKEN_RE.findall(doc_block)) - {
                    "default"
                }
                unsupported = sorted(documented_tokens - allowed_choices)
                if unsupported:
                    mismatches.append(
                        f"{file_path.relative_to(PROJECT_ROOT)}::{class_node.name}: "
                        f"parameter '{param_name}' documents unsupported choices "
                        f"{unsupported}; "
                        f"allowed choices are {sorted(allowed_choices)}"
                    )

            if (
                _allows_none(value_node)
                and "none" not in doc_lower
                and "optional" not in doc_lower
            ):
                mismatches.append(
                    f"{file_path.relative_to(PROJECT_ROOT)}::{class_node.name}: "
                    f"parameter '{param_name}' allows None in _parameter_constraints "
                    f"but docs do not mention None/optional"
                )

    assert not mismatches, "\n" + "\n".join(sorted(mismatches))
