"""
examples/agent_tx_demo.py — SagaLog: compensation-safe agent tool calls (RP-7)

Demonstrates how @compensable + run_saga provide rollback for an agent
workflow that mutates external state (a mock database + file system).

Scenario:
  A 4-step agent workflow writes a user record, uploads a file, queues a
  notification, and then fails at step 4 (schema validation error).
  Steps 1-3 have already mutated external state.  run_saga() automatically
  calls their compensation functions in reverse order, leaving the system
  in its original state.

Run: python -m examples.agent_tx_demo
"""

import os
import tempfile
from patterns.agent_tx import SagaLog, compensable, run_saga

# ---------------------------------------------------------------------------
# Simulated external state
# ---------------------------------------------------------------------------

_users_db: dict = {}
_uploaded_files: list = []
_notification_queue: list = []


def _reset_state():
    _users_db.clear()
    _uploaded_files.clear()
    _notification_queue.clear()


# ---------------------------------------------------------------------------
# Tool compensations (the "undo" halves)
# ---------------------------------------------------------------------------

def _undo_db_insert(result: str, user_id: str, email: str) -> None:
    _users_db.pop(user_id, None)
    print(f"  [undo] removed user {user_id!r} from DB")


def _undo_file_upload(result: str, path: str, content: bytes) -> None:
    if path in _uploaded_files:
        _uploaded_files.remove(path)
    print(f"  [undo] removed upload {os.path.basename(path)!r}")


def _undo_queue_notification(result: str, user_id: str, message: str) -> None:
    entry = (user_id, message)
    if entry in _notification_queue:
        _notification_queue.remove(entry)
    print(f"  [undo] dequeued notification for {user_id!r}")


# ---------------------------------------------------------------------------
# Compensable tools
# ---------------------------------------------------------------------------

@compensable(_undo_db_insert)
def db_insert_user(user_id: str, email: str) -> str:
    _users_db[user_id] = email
    print(f"  [tool] db_insert_user({user_id!r}) → ok")
    return f"inserted:{user_id}"


@compensable(_undo_file_upload)
def upload_file(path: str, content: bytes) -> str:
    _uploaded_files.append(path)
    print(f"  [tool] upload_file({os.path.basename(path)!r}) → ok")
    return f"uploaded:{path}"


@compensable(_undo_queue_notification)
def queue_notification(user_id: str, message: str) -> str:
    _notification_queue.append((user_id, message))
    print(f"  [tool] queue_notification({user_id!r}) → ok")
    return f"queued:{user_id}"


def validate_schema(payload: dict) -> bool:
    """Step 4 — no compensation needed (pure read, no side effects)."""
    if "required_field" not in payload:
        raise ValueError(f"schema validation failed: missing 'required_field' in {payload}")
    return True


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _separator(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f" {title}")
    print('─' * 50)


def demo_happy_path() -> None:
    _separator("Scenario 1 — happy path (no rollback needed)")
    _reset_state()
    saga = SagaLog()

    with run_saga(saga):
        db_insert_user("u1", "alice@example.com", _saga=saga)
        upload_file("/reports/u1_report.pdf", b"data", _saga=saga)
        queue_notification("u1", "Your report is ready", _saga=saga)
        validate_schema({"required_field": "present"})

    print(f"\n  DB: {_users_db}")
    print(f"  Uploads: {[os.path.basename(p) for p in _uploaded_files]}")
    print(f"  Queue: {_notification_queue}")
    print(f"  SagaLog entries: {len(saga)}")


def demo_partial_failure() -> None:
    _separator("Scenario 2 — failure at step 4 (steps 1-3 rolled back)")
    _reset_state()
    saga = SagaLog()

    print("\n  State before agent run:")
    print(f"    DB: {_users_db}, Uploads: {_uploaded_files}, Queue: {_notification_queue}")

    print("\n  Running agent workflow...")
    try:
        with run_saga(saga):
            db_insert_user("u2", "bob@example.com", _saga=saga)
            upload_file("/reports/u2_report.pdf", b"data", _saga=saga)
            queue_notification("u2", "Your report is ready", _saga=saga)
            validate_schema({"wrong_field": "oops"})  # raises ValueError
    except ValueError as exc:
        print(f"\n  [agent] failed: {exc}")

    print(f"\n  State after rollback (should be empty):")
    print(f"    DB: {_users_db}")
    print(f"    Uploads: {_uploaded_files}")
    print(f"    Queue: {_notification_queue}")

    assert _users_db == {}, "DB should be empty after rollback"
    assert _uploaded_files == [], "Upload list should be empty after rollback"
    assert _notification_queue == [], "Notification queue should be empty after rollback"
    print("\n  All assertions pass — system is clean.")


def demo_sqlite_backend() -> None:
    _separator("Scenario 3 — SQLiteSagaBackend (crash-durable audit trail)")
    from patterns.agent_tx import SQLiteSagaBackend

    _reset_state()
    backend = SQLiteSagaBackend()
    saga = SagaLog(backend=backend)

    try:
        with run_saga(saga):
            db_insert_user("u3", "carol@example.com", _saga=saga)
            upload_file("/reports/u3_report.pdf", b"data", _saga=saga)
            raise RuntimeError("simulated crash mid-workflow")
    except RuntimeError:
        pass

    audit = backend.recorded_tool_names()
    print(f"\n  Durable audit log of executed tools: {audit}")
    print(f"  (These tool names survive a process restart if backed by on-disk SQLite)")
    assert audit == ["db_insert_user", "upload_file"]
    print("  Assertion pass — audit trail correct.")


if __name__ == "__main__":
    demo_happy_path()
    demo_partial_failure()
    demo_sqlite_backend()
    print("\n=== all demo scenarios complete ===")
