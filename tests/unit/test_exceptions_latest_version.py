from io import StringIO

from papairus.exceptions import NoChangesWarning


def test_no_changes_warning_writes_to_default_stream(monkeypatch):
    """
    Test that the NoChangesWarning class writes the warning message to the default output stream.
    
    Args:
        monkeypatch: A pytest monkeypatch object.
    
    Returns:
        None
    """
    stream = StringIO()
    monkeypatch.setattr("click.get_text_stream", lambda _name: stream)

    warning = NoChangesWarning("nothing to commit")
    warning.show()

    assert "Warning: nothing to commit" in stream.getvalue()


def test_no_changes_warning_respects_custom_stream():
    """
    Test that the NoChangesWarning class respects the custom stream.
    
    Returns:
        None
    """
    stream = StringIO()
    warning = NoChangesWarning("custom target")

    warning.show(stream)

    assert "Warning: custom target" in stream.getvalue()
