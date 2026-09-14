# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for ExampleHiddenStatesConnector abort handling.

Covers the abort path where a request is cancelled while still waiting (never
scheduled): ``request_finished`` must not raise, and the worker must not report
a stale ``finished_sending`` completion for a request the scheduler already
freed.
"""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
)


def _bare_connector() -> ExampleHiddenStatesConnector:
    """Instance with scheduler-side state but bypassing ``__init__`` (no engine)."""
    conn = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    conn._request_filenames = {}
    conn._pending_saves = {}
    conn._lock_fds = {}
    conn._cache_kv_group_id = 1
    conn._connector_metadata = None
    conn._req_copy_events = {}
    conn._accumulated_finished_req_ids = set()
    return conn


def test_get_finished_count_is_one():
    # Only TP rank 0 writes, so KVOutputAggregator must expect a single
    # finished_sending notification per request (not the TP world size).
    assert _bare_connector().get_finished_count() == 1


def test_request_finished_is_noop_for_never_scheduled_request():
    # A request aborted while still queued never reaches build_connector_meta,
    # so no filename was recorded. request_finished must not raise KeyError.
    conn = _bare_connector()
    request = SimpleNamespace(request_id="cmpl-aborted", kv_transfer_params=None)
    assert conn.request_finished(request, []) == (False, None)


def test_request_finished_all_groups_handles_missing_group():
    # Guard against indexing a nonexistent per-group block table.
    conn = _bare_connector()
    conn._cache_kv_group_id = 2
    request = SimpleNamespace(request_id="cmpl-aborted", kv_transfer_params=None)
    assert conn.request_finished_all_groups(request, ([], [])) == (False, None)


def test_get_finished_does_not_report_untracked_request():
    # A never-scheduled aborted request has no copy event. get_finished must not
    # report it as done_sending, or the scheduler asserts it is still tracked.
    conn = _bare_connector()
    assert conn.get_finished({"cmpl-aborted"}) == (None, None)
    assert conn._accumulated_finished_req_ids == set()


def test_get_finished_reports_tracked_completed_request():
    conn = _bare_connector()

    class _DoneEvent:
        def query(self) -> bool:
            return True

    conn._req_copy_events["cmpl-done"] = _DoneEvent()
    done_sending, done_recving = conn.get_finished({"cmpl-done"})
    assert done_sending == {"cmpl-done"}
    assert done_recving is None
