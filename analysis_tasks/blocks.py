import hashlib
import json
from archive import atomic
from analysis_tasks.catalog import TaskIdConflict
from runtime_models.schema import ModelRef
from runtime_models.registry import resolve_model
from duration_inference.segments import phase_space
from duration_inference.blocksum import joint_shift_moments, summary_evidence
from duration_inference.evidence import ZeroEvidenceError, InferenceNumericalError
from observations.blocks import parse_block
from run_reports.publication import PublicationError, canonical_report
from run_reports.reader import ReportNotPublished, ArchiveIntegrityError
from run_reports.validation import (validate_joint_statistics,
                                    validate_shift_conservation)

def initialize_block_archive(connection):
    connection.execute("""CREATE TABLE IF NOT EXISTS block_reports(
        report_id TEXT PRIMARY KEY, input_id TEXT NOT NULL, body TEXT NOT NULL)""")
    connection.execute("""CREATE TABLE IF NOT EXISTS block_receipts(
        report_id TEXT PRIMARY KEY REFERENCES block_reports(report_id),
        receipt_id TEXT UNIQUE NOT NULL, body TEXT NOT NULL)""")

def shift_dimensions(states):
    dimensions = []
    for shift in (0, 1):
        dimensions.extend([shift, "occupancy", state] for state in states)
        dimensions.append([shift, "renewals", ""])
    return dimensions

def request_identity(report_id, left_block_id, right_block_id):
    content = {"report_id": report_id,
               "blocks": [left_block_id, right_block_id]}
    return hashlib.sha256(json.dumps(
        content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _load_block_report(connection, report_id):
    row = connection.execute(
        "SELECT input_id,body FROM block_reports WHERE report_id=?",
        (report_id,)).fetchone()
    receipt = connection.execute(
        "SELECT receipt_id,body FROM block_receipts WHERE report_id=?",
        (report_id,)).fetchone()
    if row is None and receipt is None:
        return None
    if row is None or receipt is None:
        raise ArchiveIntegrityError("incomplete block publication")
    return row[0], row[1], receipt[0], receipt[1]

def _check_model_binding(model, phases, parsed):
    if list(model.states) != list(parsed.states):
        raise ValueError("block states do not match the model")
    if ([[phase.state, phase.remaining] for phase in phases]
            != [list(entry) for entry in parsed.phases]):
        raise ValueError("block phases do not match the model")
    try:
        evidence = summary_evidence(model, phases, parsed.head)
    except ZeroEvidenceError as error:
        raise ValueError("block evidence does not match its content") from error
    if (evidence.mantissa != parsed.evidence["mantissa"]
            or evidence.exponent != parsed.evidence["exponent"]):
        raise ValueError("block evidence does not match its content")

def sign_blocks(center, report_id, left, right, *, before_commit=None):
    if not isinstance(report_id, str) or not report_id:
        raise ValueError("report ID must be a nonempty string")
    left_block = parse_block(left)
    right_block = parse_block(right)
    if left_block.run_id != right_block.run_id:
        raise ValueError("blocks belong to different runs")
    if left_block.model_ref != right_block.model_ref:
        raise ValueError("blocks reference different models")
    if left_block.start + left_block.length != right_block.start:
        raise ValueError("blocks are not adjacent in run order")
    revision = resolve_model(center, ModelRef(*left_block.model_ref))
    model = revision.definition
    phases = phase_space(model)
    _check_model_binding(model, phases, left_block)
    _check_model_binding(model, phases, right_block)
    input_id = request_identity(report_id, left_block.block_id, right_block.block_id)
    stored = _load_block_report(center, report_id)
    if stored is not None:
        if stored[0] != input_id:
            raise TaskIdConflict(report_id)
        return read_blocks(center, report_id)
    evidence, mean, covariance = joint_shift_moments(
        model, phases, left_block.head, right_block.tail)
    states = list(model.states)
    dimensions = shift_dimensions(states)
    validate_joint_statistics(mean, covariance)
    lengths = (left_block.length, right_block.length)
    validate_shift_conservation(states, lengths, mean, covariance)
    body = {
        "report_id": report_id,
        "input_id": input_id,
        "run_id": left_block.run_id,
        "model_ref": list(left_block.model_ref),
        "blocks": [left_block.header, right_block.header],
        "states": states,
        "dimensions": dimensions,
        "evidence": evidence.to_data(),
        "mean": mean,
        "covariance": covariance,
    }
    _publish_block_report(center, report_id, input_id, body,
                          before_commit=before_commit)
    return read_blocks(center, report_id)

def _publish_block_report(center, report_id, input_id, body, *, before_commit=None):
    encoded = canonical_report(body)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    receipt_id = hashlib.sha256((input_id + ":" + digest).encode()).hexdigest()
    receipt = {"receipt_id": receipt_id, "report_id": report_id,
               "input_id": input_id, "report_sha256": digest}
    try:
        with atomic(center):
            stored = _load_block_report(center, report_id)
            if stored is not None:
                if stored[0] != input_id:
                    raise TaskIdConflict(report_id)
                return
            center.execute("INSERT INTO block_reports VALUES(?,?,?)",
                           (report_id, input_id, encoded))
            center.execute("INSERT INTO block_receipts VALUES(?,?,?)",
                           (report_id, receipt_id, canonical_report(receipt)))
            if before_commit is not None:
                before_commit()
    except TaskIdConflict:
        raise
    except Exception as error:
        raise PublicationError("block report publication rolled back") from error

def read_blocks(center, report_id):
    if not isinstance(report_id, str) or not report_id:
        raise ValueError("report ID must be a nonempty string")
    with atomic(center):
        stored = _load_block_report(center, report_id)
        if stored is None:
            raise ReportNotPublished(report_id)
        input_id, encoded, receipt_id, receipt_encoded = stored
        try:
            body = json.loads(encoded)
            receipt = json.loads(receipt_encoded)
            digest = hashlib.sha256(canonical_report(body).encode()).hexdigest()
            expected_id = hashlib.sha256(
                (input_id + ":" + digest).encode()).hexdigest()
            expected = {"receipt_id": expected_id, "report_id": report_id,
                        "input_id": input_id, "report_sha256": digest}
            if receipt != expected or receipt_id != expected_id:
                raise ArchiveIntegrityError("block receipt mismatch")
            if (not isinstance(body, dict) or body["report_id"] != report_id
                    or body["input_id"] != input_id):
                raise ArchiveIntegrityError("block report identity mismatch")
            headers = body["blocks"]
            if request_identity(report_id, headers[0]["block_id"],
                                headers[1]["block_id"]) != input_id:
                raise ArchiveIntegrityError("block request mismatch")
            states = body["states"]
            if body["dimensions"] != shift_dimensions(states):
                raise ArchiveIntegrityError("block dimension mismatch")
            if len(body["mean"]) != len(body["dimensions"]):
                raise ArchiveIntegrityError("block report dimension mismatch")
            lengths = (headers[0]["length"], headers[1]["length"])
            validate_joint_statistics(body["mean"], body["covariance"])
            validate_shift_conservation(states, lengths,
                                        body["mean"], body["covariance"])
        except (TypeError, KeyError, IndexError, ValueError,
                ArithmeticError) as error:
            raise ArchiveIntegrityError("invalid archived block report") from error
        body["receipt"] = receipt
        return body
