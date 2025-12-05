import importlib
import sys
import types


def test_gradio_interface_helpers(monkeypatch):
    class DummyContext:
        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_, **__):
            return False

    dummy_gradio = types.SimpleNamespace(
        HTML=lambda *args, **kwargs: f"HTML-{kwargs.get('label', '')}",
        Blocks=lambda *args, **kwargs: DummyContext(),
        Markdown=lambda *args, **kwargs: None,
        Tab=lambda *args, **kwargs: DummyContext(),
        Row=lambda *args, **kwargs: DummyContext(),
        Column=lambda *args, **kwargs: DummyContext(),
        Textbox=lambda *args, **kwargs: f"Textbox-{kwargs.get('label', '')}",
        Button=lambda *args, **kwargs: types.SimpleNamespace(click=lambda *a, **k: None),
        ClearButton=lambda *args, **kwargs: types.SimpleNamespace(click=lambda *a, **k: None),
        close_all=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "gradio", dummy_gradio)
    monkeypatch.setitem(sys.modules, "markdown", types.SimpleNamespace(markdown=lambda value: f"<p>{value}</p>"))

    from papairus.chat_with_repo import gradio_interface as gi

    importlib.reload(gi)
    monkeypatch.setattr(gi.GradioInterface, "setup_gradio_interface", lambda self: None)

    iface = gi.GradioInterface(lambda msg, sysmsg: (msg, "resp", "recall", "kw", "code", "codex"))

    wrapper_result = iface.wrapper_respond("question", "system")
    assert "Response" in wrapper_result[1]
    assert "Embedding Recall" in wrapper_result[2]

    cleaned = iface.clean()
    assert isinstance(cleaned[1], str)
    assert isinstance(cleaned[4], str)
