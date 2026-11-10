from qtrc_stop.checkpoint import RunState


def test_run_state_roundtrip(tmp_path):
    state = RunState(tmp_path)
    state.update_task("a", "done", artifact="x")
    assert state.task_done("a")
