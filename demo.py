import json
import os
import tempfile
from archive import connect
from examples import two_state, single_state, observation_batch
from runtime_models.registry import register_model
from observations.sequences import revise_observations
from observations.blocks import issue_block
from analysis_tasks.requests import WindowRequest as W
from analysis_tasks.service import freeze, analyze
from analysis_tasks.blocks import sign_blocks, read_blocks
from run_reports.reader import read_report, ReportNotPublished
from run_reports.publication import PublicationError

def v1_demo(directory):
    path = os.path.join(directory, "archive.sqlite3")
    connection = connect(path)
    try:
        ref = register_model(connection, "machine-v1", two_state())
        old = revise_observations(connection, "machine", observation_batch("abba"))
        windows = [W(0, 3), W(1, 4)]
        freeze(connection, "frozen-before-correction", ref, old, 4, windows)
        new = revise_observations(connection, "machine", observation_batch("bbbb", 2))
        freeze(connection, "after-correction", ref, new, 4, windows)
        def fail():
            raise RuntimeError("demo controlled precommit failure")
        try:
            analyze(connection, "frozen-before-correction", before_commit=fail)
        except PublicationError:
            try:
                read_report(connection, "frozen-before-correction")
            except ReportNotPublished:
                print("Publication rolled back; frozen task remains retryable.")
    finally:
        connection.close()
    connection = connect(path)
    try:
        for task in ("frozen-before-correction", "after-correction"):
            report = analyze(connection, task)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            if read_report(connection, task) != report:
                raise AssertionError("archived report changed")
    finally:
        connection.close()

def block_demo(directory):
    model = single_state()
    day_path = os.path.join(directory, "day-station.sqlite3")
    night_path = os.path.join(directory, "night-station.sqlite3")
    center_path = os.path.join(directory, "center.sqlite3")
    day = connect(day_path)
    try:
        day_ref = register_model(day, "shift-model", model)
        day_revision = revise_observations(day, "day-line", observation_batch("xx"))
        left = issue_block(day, "run-9", 0, day_ref, day_revision, 2)
    finally:
        day.close()
    night = connect(night_path)
    try:
        night_ref = register_model(night, "shift-model", model)
        night_revision = revise_observations(night, "night-line",
                                             observation_batch("x"))
        right = issue_block(night, "run-9", 2, night_ref, night_revision, 1)
    finally:
        night.close()
    carried = [json.loads(json.dumps(piece)) for piece in (left, right)]
    center = connect(center_path)
    try:
        register_model(center, "shift-model", model)
        report = sign_blocks(center, "shift-report-1", carried[0], carried[1])
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        if sign_blocks(center, "shift-report-1", carried[0], carried[1]) != report:
            raise AssertionError("reclaimed block report changed")
        if read_blocks(center, "shift-report-1") != report:
            raise AssertionError("archived block report changed")
    finally:
        center.close()
    center = connect(center_path)
    try:
        recovered = read_blocks(center, "shift-report-1")
        if recovered != report:
            raise AssertionError("reopened center lost the final report")
        print("Center reopened; final report and receipt reclaimed "
              "under shift-report-1.")
    finally:
        center.close()

def main():
    with tempfile.TemporaryDirectory(prefix="runtime-archive-demo-") as directory:
        v1_demo(directory)
        block_demo(directory)

if __name__ == "__main__":
    main()
