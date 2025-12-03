from io import StringIO

from papairus.exceptions import NoChangesWarning


def test_no_changes_warning_writes_to_default_stream(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr("click.get_text_stream", lambda _name: stream)

    warning = NoChangesWarning("nothing to commit")
    warning.show()

    assert "Warning: nothing to commit" in stream.getvalue()


def test_no_changes_warning_respects_custom_stream():
    stream = StringIO()
    warning = NoChangesWarning("custom target")

    warning.show(stream)

    assert "Warning: custom target" in stream.getvalue()
