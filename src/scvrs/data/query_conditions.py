"""Convert structured query output into condition-verifier text fields."""

from __future__ import annotations

from typing import Any, Iterable


def _join_unique(items: Iterable[str]) -> str:
    result, seen = [], set()
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return " ; ".join(result)


def _object_text(obj: dict | None) -> str:
    return str((obj or {}).get("text", "")).strip()


def _relation_text(item: dict) -> str:
    reference = item.get("reference_object") or {}
    return f"{item.get('type', '')} {reference.get('text', '')}".strip()


def build_condition_record(parsed: dict[str, Any]) -> dict[str, Any]:
    """Build the public condition-verifier text/data contract."""
    attributes = parsed.get("attributes", [])
    positions = parsed.get("absolute_positions", [])
    relations = parsed.get("relations", [])
    comparisons = parsed.get("comparisons", [])
    quantity = parsed.get("quantity")
    subject_attributes = [
        item for item in attributes if item.get("owner") == "subject"
    ]
    reference_attributes = [
        item for item in attributes if item.get("owner") == "reference"
    ]
    subject_positions = [
        item for item in positions if item.get("owner") == "subject"
    ]
    reference_positions = [
        item for item in positions if item.get("owner") == "reference"
    ]
    reference_objects = [
        item.get("reference_object")
        for item in relations + comparisons
        if item.get("reference_object")
    ]
    reference_objects.extend(
        item.get("owner_object")
        for item in reference_attributes + reference_positions
        if item.get("owner_object")
    )
    target = _object_text(parsed.get("object"))
    quantity_text = str((quantity or {}).get("text", "")).strip()
    subject_attribute_text = _join_unique(
        item.get("text", "") for item in subject_attributes
    )
    reference_attribute_text = _join_unique(
        item.get("text", "") for item in reference_attributes
    )
    subject_position_text = _join_unique(
        item.get("text", "") for item in subject_positions
    )
    reference_position_text = _join_unique(
        item.get("text", "") for item in reference_positions
    )
    relation_text = _join_unique(_relation_text(item) for item in relations)
    comparison_text = _join_unique(
        _relation_text(item) for item in comparisons
    )
    reference_object_text = _join_unique(
        _object_text(item) for item in reference_objects
    )
    prompt_fields = (
        ("OBJECT", target),
        ("QUANTITY", quantity_text),
        ("SUBJECT_ATTRIBUTE", subject_attribute_text),
        ("SUBJECT_POSITION", subject_position_text),
        ("RELATION", relation_text),
        ("REFERENCE_OBJECT", reference_object_text),
        ("REFERENCE_ATTRIBUTE", reference_attribute_text),
        ("REFERENCE_POSITION", reference_position_text),
        ("COMPARISON", comparison_text),
    )
    return {
        "query": parsed.get("context", ""),
        "conditions": {
            "object": parsed.get("object"),
            "quantity": quantity,
            "subject_attributes": subject_attributes,
            "reference_attributes": reference_attributes,
            "subject_positions": subject_positions,
            "reference_positions": reference_positions,
            "relations": relations,
            "comparisons": comparisons,
        },
        "text_inputs": {
            "context_text": parsed.get("context", ""),
            "object_text": target,
            "quantity_text": quantity_text,
            "subject_attribute_text": subject_attribute_text,
            "subject_position_text": subject_position_text,
            "relation_text": relation_text,
            "reference_object_text": reference_object_text,
            "reference_attribute_text": reference_attribute_text,
            "reference_position_text": reference_position_text,
            "comparison_text": comparison_text,
            "condition_prompt": " ".join(
                f"[{name}] {value}"
                for name, value in prompt_fields
                if value
            ),
        },
    }
