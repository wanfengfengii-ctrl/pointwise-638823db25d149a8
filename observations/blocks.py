from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from archive import atomic
from runtime_models.registry import resolve_model
from observations.sequences import read_sequence_revision
from observations.snapshots import seal_prefix
from duration_inference.blocksum import (SUMMARY_FORMAT, block_summary,
                                         summary_evidence)

HEADER_KEYS = ("block_id", "run_id", "start", "length", "model_ref",
               "sequence_ref", "snapshot_id")

def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)

def block_identity(content):
    return hashlib.sha256(_canonical(content).encode()).hexdigest()

def block_header(block):
    return {key: block[key] for key in HEADER_KEYS}

def _encode_summary(summary):
    return {
        "kernel": [[str(x) for x in row] for row in summary["kernel"]],
        "first": [[[str(x) for x in vector] for vector in row]
                  for row in summary["first"]],
        "second": [[[[str(x) for x in vector] for vector in matrix]
                    for matrix in row] for row in summary["second"]],
    }

def issue_block(source, run_id, start, model_ref, sequence_revision, prefix):
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run ID must be a nonempty string")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if type(prefix) is not int or prefix < 1:
        raise ValueError("prefix must be a positive integer")
    with atomic(source):
        revision = resolve_model(source, model_ref)
        records = read_sequence_revision(source, sequence_revision)
        if prefix > len(records):
            raise ValueError("prefix exceeds the sequence length")
        snapshot = seal_prefix(source, sequence_revision, prefix)
        model = revision.definition
        symbols = tuple(record.symbol for record in snapshot.records)
        phases, head = block_summary(model, symbols, "head")
        _, tail = block_summary(model, symbols, "tail")
        evidence = summary_evidence(model, phases, head)
        body = {
            "format": SUMMARY_FORMAT,
            "states": list(model.states),
            "phases": [[phase.state, phase.remaining] for phase in phases],
            "evidence": evidence.to_data(),
            "head": _encode_summary(head),
            "tail": _encode_summary(tail),
        }
        content = {
            "run_id": run_id,
            "start": start,
            "length": snapshot.prefix,
            "model_ref": [revision.ref.revision_id, revision.ref.digest],
            "sequence_ref": [snapshot.sequence_id, snapshot.version],
            "snapshot_id": snapshot.snapshot_id,
            "body": body,
        }
        block = {"block_id": block_identity(content)}
        block.update(content)
        return block

@dataclass(frozen=True)
class ParsedBlock:
    block_id: str
    run_id: str
    start: int
    length: int
    model_ref: tuple
    sequence_ref: tuple
    snapshot_id: str
    states: tuple
    phases: tuple
    evidence: dict
    head: dict
    tail: dict
    header: dict

def _parse_evidence(raw):
    if not isinstance(raw, dict):
        raise ValueError("invalid block evidence")
    try:
        mantissa = raw["mantissa"]
        exponent = raw["exponent"]
        log_evidence = raw["log_evidence"]
        probability = raw["probability"]
        representable = raw["probability_representable"]
    except KeyError as error:
        raise ValueError("block evidence incomplete") from error
    if not isinstance(mantissa, str):
        raise ValueError("invalid evidence mantissa")
    try:
        parsed = Decimal(mantissa)
    except InvalidOperation as error:
        raise ValueError("invalid evidence mantissa") from error
    if not parsed.is_finite() or not Decimal(1) <= parsed < Decimal(10):
        raise ValueError("invalid evidence mantissa")
    if type(exponent) is not int:
        raise ValueError("invalid evidence exponent")
    if (isinstance(log_evidence, bool) or not isinstance(log_evidence, (int, float))
            or not math.isfinite(log_evidence)):
        raise ValueError("invalid log evidence")
    if probability is not None and (
            isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or probability <= 0):
        raise ValueError("invalid evidence probability")
    if type(representable) is not bool or representable != (probability is not None):
        raise ValueError("invalid evidence representable flag")
    return {"mantissa": mantissa, "exponent": exponent,
            "log_evidence": log_evidence, "probability": probability,
            "probability_representable": representable}

def _decode_decimal(value):
    if not isinstance(value, str):
        raise ValueError("summary entries must be decimal strings")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal in summary") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("invalid decimal in summary")
    return parsed

def _decode_array(data, shape):
    if not isinstance(data, list) or len(data) != shape[0]:
        raise ValueError("summary shape mismatch")
    if len(shape) == 1:
        return [_decode_decimal(entry) for entry in data]
    return [_decode_array(entry, shape[1:]) for entry in data]

def _decode_summary(raw, count, dims):
    if not isinstance(raw, dict):
        raise ValueError("invalid block summary")
    try:
        kernel = raw["kernel"]
        first = raw["first"]
        second = raw["second"]
    except KeyError as error:
        raise ValueError("block summary incomplete") from error
    return {"kernel": _decode_array(kernel, (count, count)),
            "first": _decode_array(first, (count, count, dims)),
            "second": _decode_array(second, (count, count, dims, dims))}

def parse_block(raw):
    if not isinstance(raw, dict):
        raise ValueError("block piece must be a mapping")
    try:
        header = {key: raw[key] for key in HEADER_KEYS}
        body = raw["body"]
    except KeyError as error:
        raise ValueError("block piece incomplete") from error
    block_id = header["block_id"]
    run_id = header["run_id"]
    start = header["start"]
    length = header["length"]
    model_ref = header["model_ref"]
    sequence_ref = header["sequence_ref"]
    snapshot_id = header["snapshot_id"]
    if not isinstance(block_id, str) or not block_id:
        raise ValueError("block ID must be a nonempty string")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run ID must be a nonempty string")
    if type(start) is not int or start < 0:
        raise ValueError("invalid block start")
    if type(length) is not int or length < 1:
        raise ValueError("invalid block length")
    if (not isinstance(model_ref, (list, tuple)) or len(model_ref) != 2
            or any(not isinstance(part, str) or not part for part in model_ref)):
        raise ValueError("invalid block model reference")
    if (not isinstance(sequence_ref, (list, tuple)) or len(sequence_ref) != 2
            or not isinstance(sequence_ref[0], str) or not sequence_ref[0]
            or type(sequence_ref[1]) is not int or sequence_ref[1] < 1):
        raise ValueError("invalid block sequence reference")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("invalid block snapshot reference")
    content = {"run_id": run_id, "start": start, "length": length,
               "model_ref": [model_ref[0], model_ref[1]],
               "sequence_ref": [sequence_ref[0], sequence_ref[1]],
               "snapshot_id": snapshot_id, "body": body}
    if block_identity(content) != block_id:
        raise ValueError("block content identity mismatch")
    if not isinstance(body, dict) or body.get("format") != SUMMARY_FORMAT:
        raise ValueError("unsupported block body")
    states = body.get("states")
    if (not isinstance(states, list) or not states
            or any(not isinstance(state, str) or not state for state in states)
            or len(set(states)) != len(states)):
        raise ValueError("invalid block states")
    raw_phases = body.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("invalid block phases")
    phases = []
    for entry in raw_phases:
        if (not isinstance(entry, (list, tuple)) or len(entry) != 2
                or type(entry[0]) is not int or entry[0] < 0
                or type(entry[1]) is not int or entry[1] < 1):
            raise ValueError("invalid block phase entry")
        phases.append((entry[0], entry[1]))
    evidence = _parse_evidence(body.get("evidence"))
    dims = len(states) + 1
    head = _decode_summary(body.get("head"), len(phases), dims)
    tail = _decode_summary(body.get("tail"), len(phases), dims)
    header = {"block_id": block_id, "run_id": run_id, "start": start,
              "length": length, "model_ref": [model_ref[0], model_ref[1]],
              "sequence_ref": [sequence_ref[0], sequence_ref[1]],
              "snapshot_id": snapshot_id}
    return ParsedBlock(block_id, run_id, start, length,
                       (model_ref[0], model_ref[1]),
                       (sequence_ref[0], sequence_ref[1]), snapshot_id,
                       tuple(states), tuple(phases), evidence, head, tail, header)
