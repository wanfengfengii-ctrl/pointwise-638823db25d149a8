import json
import math
import os
import tempfile
import unittest
from archive import connect
from examples import single_state, two_state, observation_batch
from runtime_models.schema import ModelDefinition, ModelRef
from runtime_models.registry import register_model, UnknownModelReference
from observations.sequences import revise_observations, SequenceRevision
from observations.blocks import issue_block, block_identity
from observations.snapshots import SnapshotReferenceError
from analysis_tasks.blocks import sign_blocks, read_blocks
from analysis_tasks.catalog import TaskIdConflict
from analysis_tasks.requests import WindowRequest as W
from analysis_tasks.service import freeze, analyze
from duration_inference.evidence import ZeroEvidenceError
from run_reports.reader import ReportNotPublished, ArchiveIntegrityError
from run_reports.publication import PublicationError
from tests.oracle import enumerate_paths, assert_report

ALTERNATING = ModelDefinition(
    ["idle", "run"], [[0, 1], [1, 0]],
    [[(1, 1)], [(1, 1)]], [[("a", 1)], [("b", 1)]])
RARE = ModelDefinition(
    ["run"], [[1]], [[(2, 1)]], [[("x", 1e-10), ("z", 0.9999999999)]])

def issue_pair(model, left_symbols, right_symbols, run_id="run-1", start=0,
               revision_id="shift-model"):
    left_connection = connect()
    left_ref = register_model(left_connection, revision_id, model)
    left_version = revise_observations(
        left_connection, "day-line", observation_batch(left_symbols))
    left = issue_block(left_connection, run_id, start, left_ref, left_version,
                       len(left_symbols))
    left_connection.close()
    right_connection = connect()
    right_ref = register_model(right_connection, revision_id, model)
    right_version = revise_observations(
        right_connection, "night-line", observation_batch(right_symbols))
    right = issue_block(right_connection, run_id, start + len(left_symbols),
                        right_ref, right_version, len(right_symbols))
    right_connection.close()
    carried = (json.loads(json.dumps(left)), json.loads(json.dumps(right)))
    assert carried[0] == left and carried[1] == right
    return carried

def sign_pair(model, left, right, report_id="shift-1", revision_id="shift-model"):
    center = connect()
    register_model(center, revision_id, model)
    try:
        return sign_blocks(center, report_id, left, right)
    finally:
        center.close()

class TestShiftBlocks(unittest.TestCase):
    def test_renewal_joint_report(self):
        model = single_state()
        left, right = issue_pair(model, "xx", "x")
        center = connect()
        register_model(center, "shift-model", model)
        report = sign_blocks(center, "shift-1", left, right)
        self.assertEqual(report["report_id"], "shift-1")
        self.assertEqual(report["run_id"], "run-1")
        self.assertEqual(report["states"], ["run"])
        self.assertEqual(report["dimensions"],
                         [[0, "occupancy", "run"], [0, "renewals", ""],
                          [1, "occupancy", "run"], [1, "renewals", ""]])
        self.assertEqual(report["mean"], [2, 0.5, 1, 0.5])
        self.assertEqual(report["covariance"][1][3],
                         report["covariance"][3][1])
        self.assertAlmostEqual(report["covariance"][1][3], -0.25, delta=1e-8)
        self.assertEqual(report["evidence"]["probability"], 1)
        assert_report(self, report,
                      enumerate_paths(model, "xxx", [W(0, 2), W(2, 3)]))
        headers = [dict((key, left[key]) for key in
                        ("block_id", "run_id", "start", "length", "model_ref",
                         "sequence_ref", "snapshot_id")),
                   dict((key, right[key]) for key in
                        ("block_id", "run_id", "start", "length", "model_ref",
                         "sequence_ref", "snapshot_id"))]
        self.assertEqual(report["blocks"], headers)
        self.assertEqual(report["receipt"]["report_id"], "shift-1")
        self.assertTrue(report["receipt"]["receipt_id"])
        self.assertEqual(read_blocks(center, "shift-1"), report)
        self.assertEqual(sign_blocks(center, "shift-1", left, right), report)
        self.assertEqual(report["receipt"],
                         read_blocks(center, "shift-1")["receipt"])
        center.close()

    def test_supported_and_unsupported_seam(self):
        left, right = issue_pair(ALTERNATING, "ab", "a")
        report = sign_pair(ALTERNATING, left, right)
        self.assertEqual(report["evidence"]["probability"], 0.5)
        self.assertEqual(report["mean"][-1], 1)
        assert_report(self, report,
                      enumerate_paths(ALTERNATING, "aba", [W(0, 2), W(2, 3)]))
        bad_left, same_right = issue_pair(ALTERNATING, "ba", "a", run_id="run-2")
        center = connect()
        register_model(center, "shift-model", ALTERNATING)
        with self.assertRaises(ZeroEvidenceError):
            sign_blocks(center, "shift-2", bad_left, same_right)
        with self.assertRaises(ReportNotPublished):
            read_blocks(center, "shift-2")
        for table in ("block_reports", "block_receipts"):
            self.assertEqual(center.execute(
                "SELECT count(*) FROM " + table).fetchone()[0], 0)
        center.close()

    def test_tiny_evidence_signs(self):
        left, right = issue_pair(RARE, "x" * 20, "x" * 20)
        report = sign_pair(RARE, left, right)
        evidence = report["evidence"]
        self.assertIsNone(evidence["probability"])
        self.assertFalse(evidence["probability_representable"])
        self.assertEqual(evidence["exponent"], -400)
        self.assertAlmostEqual(evidence["log_evidence"],
                               -400 * math.log(10), delta=1e-8)
        self.assertEqual(report["mean"], [20, 9.5, 20, 10])
        _, mean, covariance = enumerate_paths(
            RARE, "x" * 40, [W(0, 20), W(20, 40)])
        for actual, expected in zip(report["mean"], mean):
            self.assertAlmostEqual(actual, float(expected), delta=1e-8)
        for actual_row, expected_row in zip(report["covariance"], covariance):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, float(expected), delta=1e-8)

    def test_noisy_joint_matches_oracle(self):
        model = two_state()
        left, right = issue_pair(model, "abb", "baabb")
        report = sign_pair(model, left, right)
        assert_report(self, report, enumerate_paths(
            model, "abb" + "baabb", [W(0, 3), W(3, 8)]))
        self.assertEqual(report["dimensions"],
                         [[0, "occupancy", "idle"], [0, "occupancy", "run"],
                          [0, "renewals", ""], [1, "occupancy", "idle"],
                          [1, "occupancy", "run"], [1, "renewals", ""]])

    def test_nonzero_left_start(self):
        model = single_state()
        left, right = issue_pair(model, "xx", "x", run_id="run-9", start=5)
        self.assertEqual(left["start"], 5)
        self.assertEqual(right["start"], 7)
        shifted = sign_pair(model, left, right, report_id="shift-9")
        base_left, base_right = issue_pair(model, "xx", "x", run_id="run-9")
        base = sign_pair(model, base_left, base_right, report_id="shift-0")
        self.assertEqual(shifted["mean"], base["mean"])
        self.assertEqual(shifted["covariance"], base["covariance"])
        self.assertEqual(shifted["evidence"], base["evidence"])
        self.assertNotEqual(shifted["input_id"], base["input_id"])

    def test_issue_block_validation(self):
        model = single_state()
        connection = connect()
        ref = register_model(connection, "shift-model", model)
        version = revise_observations(connection, "line", observation_batch("xx"))
        for run_id in ("", 7, None):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                issue_block(connection, run_id, 0, ref, version, 2)
        for start in (True, -1, "0"):
            with self.subTest(start=start), self.assertRaises(ValueError):
                issue_block(connection, "run-1", start, ref, version, 2)
        for prefix in (True, 0, -1, 2.5):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                issue_block(connection, "run-1", 0, ref, version, prefix)
        with self.assertRaises(ValueError):
            issue_block(connection, "run-1", 0, ref, version, 3)
        with self.assertRaises(UnknownModelReference):
            issue_block(connection, "run-1", 0, ModelRef("shift-model", "bad"),
                        version, 2)
        with self.assertRaises(UnknownModelReference):
            issue_block(connection, "run-1", 0, ["shift-model", "bad"], version, 2)
        with self.assertRaises(LookupError):
            issue_block(connection, "run-1", 0, ref, SequenceRevision("line", 9), 2)
        first = issue_block(connection, "run-1", 0, ref, version, 2)
        self.assertEqual(first, issue_block(connection, "run-1", 0, ref, version, 2))
        zero = revise_observations(connection, "zero", observation_batch("zz"))
        with self.assertRaises(ZeroEvidenceError):
            issue_block(connection, "run-1", 0, ref, zero, 2)
        connection.close()

    def test_sign_block_validation(self):
        model = single_state()
        left, right = issue_pair(model, "xx", "x")
        center = connect()
        register_model(center, "shift-model", model)
        for report_id in ("", 5, None):
            with self.subTest(report_id=report_id), self.assertRaises(ValueError):
                sign_blocks(center, report_id, left, right)
        with self.assertRaises(ValueError):
            sign_blocks(center, "bad", None, right)
        with self.assertRaises(ValueError):
            sign_blocks(center, "bad", {"block_id": "x"}, right)
        tampered = json.loads(json.dumps(left))
        tampered["body"]["head"]["kernel"][0][0] = "0.75"
        with self.assertRaises(ValueError):
            sign_blocks(center, "bad", tampered, right)
        forged = json.loads(json.dumps(left))
        forged["body"]["evidence"]["mantissa"] = "9.5"
        content = dict((key, forged[key]) for key in
                       ("run_id", "start", "length", "model_ref",
                        "sequence_ref", "snapshot_id", "body"))
        forged["block_id"] = block_identity(content)
        with self.assertRaises(ValueError):
            sign_blocks(center, "bad", forged, right)
        other_run = json.loads(json.dumps(right))
        other_run["run_id"] = "run-2"
        with self.assertRaises(ValueError):
            sign_blocks(center, "bad", left, other_run)
        other_model = connect()
        other_ref = register_model(other_model, "shift-model", two_state())
        other_version = revise_observations(
            other_model, "night-line", observation_batch("a"))
        mismatched = issue_block(other_model, "run-1", 2, other_ref,
                                 other_version, 1)
        other_model.close()
        with self.assertRaises(ValueError):
            sign_blocks(center, "bad", left, mismatched)
        with self.assertRaises(ValueError):
            sign_blocks(center, "bad", right, left)
        gap = json.loads(json.dumps(right))
        gap["start"] = 3
        with self.assertRaises(ValueError):
            sign_blocks(center, "bad", left, gap)
        unknown_model = issue_pair(two_state(), "ab", "a",
                                   revision_id="other-model")
        with self.assertRaises(UnknownModelReference):
            sign_blocks(center, "bad", *unknown_model)
        center.close()

    def test_conflicting_request_rejected(self):
        model = single_state()
        left, right = issue_pair(model, "xx", "x")
        other_left, other_right = issue_pair(model, "xx", "x", run_id="run-2")
        center = connect()
        register_model(center, "shift-model", model)
        report = sign_blocks(center, "shift-1", left, right)
        with self.assertRaises(TaskIdConflict):
            sign_blocks(center, "shift-1", other_left, other_right)
        with self.assertRaises(ValueError):
            sign_blocks(center, "shift-1", right, left)
        self.assertEqual(sign_blocks(center, "shift-1", left, right), report)
        center.close()

    def test_publication_rollback(self):
        model = single_state()
        left, right = issue_pair(model, "xx", "x")
        center = connect()
        register_model(center, "shift-model", model)
        def fail():
            raise RuntimeError("controlled precommit failure")
        with self.assertRaises(PublicationError) as caught:
            sign_blocks(center, "shift-1", left, right, before_commit=fail)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        with self.assertRaises(ReportNotPublished):
            read_blocks(center, "shift-1")
        for table in ("block_reports", "block_receipts"):
            self.assertEqual(center.execute(
                "SELECT count(*) FROM " + table).fetchone()[0], 0)
        report = sign_blocks(center, "shift-1", left, right)
        self.assertEqual(read_blocks(center, "shift-1"), report)
        self.assertEqual(sign_blocks(center, "shift-1", left, right,
                                     before_commit=fail), report)
        center.close()

    def test_archive_integrity(self):
        model = single_state()
        left, right = issue_pair(model, "xx", "x")
        center = connect()
        register_model(center, "shift-model", model)
        sign_blocks(center, "shift-1", left, right)
        center.execute("DELETE FROM block_receipts WHERE report_id='shift-1'")
        with self.assertRaises(ArchiveIntegrityError):
            read_blocks(center, "shift-1")
        center.execute("DELETE FROM block_reports WHERE report_id='shift-1'")
        with self.assertRaises(ReportNotPublished):
            read_blocks(center, "shift-1")
        sign_blocks(center, "shift-1", left, right)
        center.execute("UPDATE block_reports SET body='{}' WHERE report_id='shift-1'")
        with self.assertRaises(ArchiveIntegrityError):
            read_blocks(center, "shift-1")
        center.close()

    def test_correction_and_reopen_keep_final(self):
        model = two_state()
        with tempfile.TemporaryDirectory(prefix="shift-blocks-test-") as directory:
            day_path = os.path.join(directory, "day.sqlite3")
            night_path = os.path.join(directory, "night.sqlite3")
            center_path = os.path.join(directory, "center.sqlite3")
            day = connect(day_path)
            day_ref = register_model(day, "shift-model", model)
            day_version = revise_observations(
                day, "line", observation_batch("ab"))
            left = issue_block(day, "run-1", 0, day_ref, day_version, 2)
            day.close()
            night = connect(night_path)
            night_ref = register_model(night, "shift-model", model)
            night_version = revise_observations(
                night, "line", observation_batch("a"))
            right = issue_block(night, "run-1", 2, night_ref, night_version, 1)
            night.close()
            center = connect(center_path)
            register_model(center, "shift-model", model)
            report = sign_blocks(center, "shift-1", left, right)
            center.close()
            day = connect(day_path)
            corrected = revise_observations(
                day, "line", observation_batch("bb", 2))
            corrected_left = issue_block(day, "run-1", 0, day_ref, corrected, 2)
            day.close()
            center = connect(center_path)
            with self.assertRaises(TaskIdConflict):
                sign_blocks(center, "shift-1", corrected_left, right)
            self.assertEqual(sign_blocks(center, "shift-1", left, right), report)
            recovered = read_blocks(center, "shift-1")
            self.assertEqual(recovered, report)
            self.assertEqual(recovered["mean"], report["mean"])
            center.close()

    def test_block_usable_in_any_position(self):
        model = single_state()
        connection = connect()
        ref = register_model(connection, "shift-model", model)
        version = revise_observations(connection, "line", observation_batch("x"))
        first = issue_block(connection, "run-1", 0, ref, version, 1)
        second = issue_block(connection, "run-1", 1, ref, version, 1)
        connection.close()
        report = sign_pair(model, first, second)
        assert_report(self, report,
                      enumerate_paths(model, "xx", [W(0, 1), W(1, 2)]))
        self.assertEqual(report["mean"], [1, 0, 1, 0.5])

    def test_v1_interface_coexists(self):
        model = two_state()
        left, right = issue_pair(model, "ab", "a")
        center = connect()
        ref = register_model(center, "shift-model", model)
        sign_blocks(center, "shift-1", left, right)
        version = revise_observations(center, "v1-line", observation_batch("abba"))
        freeze(center, "v1-task", ref, version, 4, [W(0, 3), W(1, 4)])
        report = analyze(center, "v1-task")
        assert_report(self, report,
                      enumerate_paths(model, "abba", [W(0, 3), W(1, 4)]))
        center.close()

if __name__ == "__main__":
    unittest.main()
