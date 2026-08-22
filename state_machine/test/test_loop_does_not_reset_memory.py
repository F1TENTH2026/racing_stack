"""Guard: loop() must not clear the opponent memory.

The memory hold silently did nothing for three real runs because an init block
was duplicated into loop(), so _last_dyn_seen_sec was set back to None on every
one of the 50 iterations per second. _opponent_memory_active() could therefore
never be true, and [OPP_MEMORY] never appeared -- while the code read correctly
in isolation and passed every unit test.

This walks the real source: whatever loop() is allowed to reset per iteration,
the memory attributes are not among them.
"""
import ast
import os

SOURCE = os.path.join(os.path.dirname(__file__), "..", "state_machine",
                      "state_machine_node.py")

# Attributes that must persist ACROSS loop iterations to do their job.
PERSISTENT = {
    "_last_dyn_seen_sec",
    "_last_dyn_gap_m",
    "_last_dyn_id",
    "_last_overtake_sec",
}


def _loop_body():
    tree = ast.parse(open(SOURCE).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "loop":
            return node
    raise AssertionError("StateMachine.loop() not found")


def _self_attrs_assigned(fn):
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                out.append((target.attr, node.lineno))
    return out


def test_loop_does_not_reset_the_opponent_memory():
    assigned = _self_attrs_assigned(_loop_body())
    offenders = [(a, ln) for a, ln in assigned if a in PERSISTENT]
    assert not offenders, (
        "loop() assigns persistent memory attribute(s) "
        + ", ".join(f"self.{a} (line {ln})" for a, ln in offenders)
        + " -- these must only be written by _update_opponent_memory()"
    )


def test_memory_attributes_are_initialised_exactly_once():
    """Once in __init__, and nowhere else outside _update_opponent_memory."""
    tree = ast.parse(open(SOURCE).read())
    init = update = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "__init__":
                init = node
            elif node.name == "_update_opponent_memory":
                update = node
    assert init is not None and update is not None

    allowed = {id(n) for fn in (init, update) for n in ast.walk(fn)}
    stray = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or id(node) in allowed:
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in PERSISTENT):
                stray.append((target.attr, node.lineno))
    assert not stray, f"memory attributes written outside __init__/_update_opponent_memory: {stray}"
