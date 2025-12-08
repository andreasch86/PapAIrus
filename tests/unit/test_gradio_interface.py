import types

from papairus.chat_with_repo import gradio_interface as gi


def test_gradio_interface_close_calls_handles(monkeypatch):
    monkeypatch.setattr(gi.GradioInterface, "setup_gradio_interface", lambda self: None)

    closed = {"demo": False, "launch": False}

    class DummyDemo:
        def close(self):
            closed["demo"] = True

    class DummyLaunch:
        def close(self):
            closed["launch"] = True

    iface = gi.GradioInterface(lambda *_args: ("", "", "", "", "", ""), {})
    iface.demo = DummyDemo()
    iface.launch_handle = DummyLaunch()

    iface.close()

    assert closed["demo"] is True
    assert closed["launch"] is True

def test_gradio_interface_wrapper_formats_output(monkeypatch):
    monkeypatch.setattr(gi.GradioInterface, "setup_gradio_interface", lambda self: None)

    iface = gi.GradioInterface(lambda *_args: ("msg", "out1", "out2", "out3", "code", "codex"), {})
    outputs = iface.wrapper_respond("msg", "system")

    assert len(outputs) == 6

def test_gradio_interface_clean_returns_defaults(monkeypatch):
    monkeypatch.setattr(gi.GradioInterface, "setup_gradio_interface", lambda self: None)

    iface = gi.GradioInterface(lambda *_args: ("", "", "", "", "", ""), {})
    outputs = iface.clean()

    assert len(outputs) == 6


def test_gradio_interface_close_handles_missing_launch(monkeypatch):
    monkeypatch.setattr(gi.GradioInterface, "setup_gradio_interface", lambda self: None)

    iface = gi.GradioInterface(lambda *_args: ("", "", "", "", "", ""), {})
    iface.demo = None
    iface.launch_handle = object()

    iface.close()
