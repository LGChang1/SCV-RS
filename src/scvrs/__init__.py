"""SCV-RS: structured candidate verification for remote sensing."""

__version__ = "0.1.0"

CONDITION_FIELDS = (
    "context_text",
    "object_text",
    "quantity_text",
    "subject_attribute_text",
    "subject_position_text",
    "relation_text",
    "reference_object_text",
    "reference_attribute_text",
    "reference_position_text",
    "comparison_text",
)

__all__ = ["CONDITION_FIELDS", "__version__"]
