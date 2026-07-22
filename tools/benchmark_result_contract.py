"""Pure validation helpers shared by benchmark import and package checking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
TOKEN_RE = re.compile(r"[A-Z0-9_]+", re.ASCII)
ARRAY_INDEX_RE = re.compile(r"0|[1-9][0-9]*", re.ASCII)


class ResultContractError(ValueError):
    """A result document does not satisfy the benchmark import contract."""


@dataclass(frozen=True)
class ResultIdentity:
    run_id: str
    method: str
    dataset: str
    splits: frozenset[str]


def require_result_document(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultContractError("result JSON must contain an object")
    return value


def parse_result_identity(result: Mapping[str, Any]) -> ResultIdentity:
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise ResultContractError("run_id must be a safe non-empty ASCII identifier")

    method = result.get("method")
    if not isinstance(method, Mapping):
        raise ResultContractError("result method must contain an object")
    method_key = method.get("key")
    if not isinstance(method_key, str) or not method_key:
        raise ResultContractError("result method key must be non-empty text")

    dataset = result.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ResultContractError("result dataset must contain an object")
    dataset_name = dataset.get("name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ResultContractError("result dataset name must be non-empty text")
    raw_splits = dataset.get("splits")
    if not isinstance(raw_splits, list) or not raw_splits:
        raise ResultContractError("result dataset splits must be a non-empty list")
    if any(not isinstance(split, str) or not split for split in raw_splits):
        raise ResultContractError("result dataset splits must contain non-empty text")

    return ResultIdentity(
        run_id=run_id,
        method=method_key,
        dataset=dataset_name,
        splits=frozenset(raw_splits),
    )


def binding_list(
    result: Mapping[str, Any], key: str, *, required: bool = False
) -> tuple[Mapping[str, Any], ...]:
    raw_bindings = result.get(key)
    if raw_bindings is None and not required:
        return ()
    if not isinstance(raw_bindings, list):
        raise ResultContractError(f"result {key} must be a list")
    bindings: list[Mapping[str, Any]] = []
    for index, binding in enumerate(raw_bindings):
        if not isinstance(binding, Mapping):
            raise ResultContractError(f"result {key}[{index}] must contain an object")
        token = binding.get("token")
        if not isinstance(token, str) or TOKEN_RE.fullmatch(token) is None:
            raise ResultContractError(f"result {key}[{index}] token is invalid")
        bindings.append(binding)
    return tuple(bindings)


def validate_registry_identity(
    identity: ResultIdentity, row: Mapping[str, str], *, token: str
) -> None:
    if row.get("method") != identity.method:
        raise ResultContractError(
            f"method mismatch for {token}: registry={row.get('method')} result={identity.method}"
        )
    if row.get("dataset") != identity.dataset or row.get("split") not in identity.splits:
        raise ResultContractError(f"dataset or split mismatch for {token}")


def _decode_pointer_part(raw_part: str, pointer: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(raw_part):
        character = raw_part[index]
        if character != "~":
            output.append(character)
            index += 1
            continue
        if index + 1 >= len(raw_part) or raw_part[index + 1] not in {"0", "1"}:
            raise ResultContractError(f"invalid JSON pointer escape: {pointer}")
        output.append("~" if raw_part[index + 1] == "0" else "/")
        index += 2
    return "".join(output)


def resolve_json_pointer(document: Any, pointer: Any) -> Any:
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ResultContractError(f"invalid JSON pointer: {pointer}")
    value = document
    for raw_part in pointer[1:].split("/"):
        part = _decode_pointer_part(raw_part, pointer)
        try:
            if isinstance(value, list):
                if ARRAY_INDEX_RE.fullmatch(part) is None:
                    raise ResultContractError(f"invalid JSON pointer array index: {pointer}")
                try:
                    array_index = int(part)
                except ValueError as error:
                    raise ResultContractError(
                        f"invalid JSON pointer array index: {pointer}"
                    ) from error
                value = value[array_index]
            elif isinstance(value, Mapping):
                value = value[part]
            else:
                raise KeyError(part)
        except (KeyError, IndexError) as error:
            raise ResultContractError(f"JSON pointer does not resolve: {pointer}") from error
    return value
