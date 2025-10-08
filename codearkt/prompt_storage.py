import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, get_args

import yaml
from jinja2 import Environment, Template

DEFAULT_BEGIN_CODE_SEQUENCE = "<execute>"
DEFAULT_END_CODE_SEQUENCE = "</execute>"
DEFAULT_BEGIN_PLAN_SEQUENCE = "<plan>"
DEFAULT_END_PLAN_SEQUENCE = "</plan>"
DEFAULT_BEGIN_FINAL_ANSWER_SEQUENCE = "<final_answer>"
DEFAULT_END_FINAL_ANSWER_SEQUENCE = "</final_answer>"
DEFAULT_EXCLUDED_STOP_SEQUENCES = ["Observation:", "Calling tools:"]
DEFAULT_INCLUDED_STOP_SEQUENCES = [DEFAULT_END_CODE_SEQUENCE]


def _schema_to_md_internal(schema: Dict[str, Any]) -> str:
    def _ref_name(ref: str) -> str:
        # "#/$defs/NestedModel" -> "NestedModel"
        parts = ref.split("/")
        return parts[-1] if parts else ref

    def _fmt_default(val: Any) -> str:
        if val is None:
            return "None"
        if isinstance(val, bool):
            return "True" if val else "False"
        if isinstance(val, (int, float)):
            return str(val)
        # strings / lists / dicts -> JSON repr
        return json.dumps(val, ensure_ascii=False)

    def _type_of(node: Any) -> str:
        if not isinstance(node, dict):
            return "object"

        # unions: anyOf
        if "anyOf" in node:
            opts = []
            for opt in node["anyOf"]:
                opts.append(_type_of(opt))
            seen = set()
            uniq = []
            for t in opts:
                if t not in seen:
                    uniq.append(t)
                    seen.add(t)
            return " | ".join(uniq)

        # refs
        if "$ref" in node:
            return _ref_name(node["$ref"])

        regular_type = node.get("type")

        # rare: list of primitive types
        if isinstance(regular_type, list):
            return " | ".join(map(str, regular_type))

        if isinstance(regular_type, str) and regular_type == "array":
            items = node.get("items", {})
            return f"array[{_type_of(items)}]"
        if regular_type:
            return str(regular_type)

        # objects without explicit type
        if "properties" in node:
            return "object"
        return "object"

    def _line_for(name: str, node: Dict[str, Any], required: bool) -> str:
        desc = (node.get("description") or "").strip()
        typ = _type_of(node)
        req = "required" if required else "optional"
        line = f"    {name}: "
        if desc:
            line += f"{desc}, "
        line += f"{typ}, {req}"
        if "default" in node:
            line += f" (default={_fmt_default(node['default'])})"
        return line

    def _render_object_block(title: str, obj: Dict[str, Any]) -> str:
        props = obj.get("properties") or {}
        req = set(obj.get("required") or [])
        lines = [f"{title}"]
        for pname, pnode in props.items():
            lines.append(_line_for(pname, pnode, pname in req))
        return "\n".join(lines)

    lines = []
    lines.append(_render_object_block("", schema))
    defs = schema.get("$defs") or {}
    for dname, dnode in defs.items():
        lines.append(_render_object_block(dname + ":", dnode))

    return "\n".join(lines).strip()


def schema_to_md(schema: Dict[str, Any]) -> str:
    try:
        return _schema_to_md_internal(schema)
    except Exception:
        return json.dumps(schema, ensure_ascii=False)


def _create_jinja_env() -> Environment:
    env = Environment(autoescape=True)
    env.filters["schema_to_md"] = schema_to_md
    return env


@dataclass
class PromptStorage:
    system: Template
    final: Template
    no_code_action: Template
    plan: Optional[Template] = None
    plan_prefix: Optional[Template] = None
    plan_suffix: Optional[Template] = None

    begin_code_sequence: str = DEFAULT_BEGIN_CODE_SEQUENCE
    end_code_sequence: str = DEFAULT_END_CODE_SEQUENCE
    begin_plan_sequence: str = DEFAULT_BEGIN_PLAN_SEQUENCE
    end_plan_sequence: str = DEFAULT_END_PLAN_SEQUENCE
    begin_final_answer_sequence: str = DEFAULT_BEGIN_FINAL_ANSWER_SEQUENCE
    end_final_answer_sequence: str = DEFAULT_END_FINAL_ANSWER_SEQUENCE
    excluded_stop_sequences: List[str] = field(
        default_factory=lambda: DEFAULT_EXCLUDED_STOP_SEQUENCES
    )
    included_stop_sequences: List[str] = field(
        default_factory=lambda: DEFAULT_INCLUDED_STOP_SEQUENCES
    )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path) as f:
            template = f.read()
        templates: Dict[str, Any] = yaml.safe_load(template)
        wrapped_templates: Dict[str, Any] = {}
        env = _create_jinja_env()
        for key, value in templates.items():
            if value is None:
                continue
            field_type = cls.__annotations__.get(key)
            if field_type is Template or Template in get_args(field_type):
                wrapped_templates[key] = env.from_string(value)
            elif field_type is str:
                wrapped_templates[key] = value.strip()
            else:
                wrapped_templates[key] = value
        obj = cls(**wrapped_templates)
        obj.excluded_stop_sequences = [s.strip() for s in obj.excluded_stop_sequences]
        obj.included_stop_sequences = [s.strip() for s in obj.included_stop_sequences]
        return obj

    @classmethod
    def default(cls) -> Self:
        current_dir = Path(__file__).parent
        return cls.load(current_dir / "prompts" / "default.yaml")
