import socket
import time
from typing import Iterator

import pytest
import requests

pytest.importorskip("gradio")
from gradio_client import Client

from papairus.chat_with_repo.gradio_interface import GradioInterface


_TEST_PORT = 7860


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_interface(url: str, timeout: float = 10) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code < 500:
                return
        except requests.RequestException:
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url} to become ready")


@pytest.fixture()
def chat_interface() -> Iterator[GradioInterface]:
    if not _port_available(_TEST_PORT):
        pytest.skip(f"Port {_TEST_PORT} is already in use; cannot start Gradio test server.")

    def _fake_respond(message: str, instruction: str):
        return (
            message,
            f"<p>Response for: {message}</p>",
            "<p>Embedding recall placeholder</p>",
            ["keyword-a", "keyword-b"],
            "<p>Code sample</p>",
            ["code-keyword-a", "code-keyword-b"],
        )

    interface = GradioInterface(
        _fake_respond,
        launch_kwargs={
            "server_name": "127.0.0.1",
            "server_port": _TEST_PORT,
            "prevent_thread_lock": True,
            "inbrowser": False,
            "show_error": True,
        },
    )
    try:
        _wait_for_interface(f"http://127.0.0.1:{_TEST_PORT}")
        yield interface
    finally:
        interface.close()


def test_gradio_endpoint_round_trip(chat_interface: GradioInterface):
    client = Client(f"http://127.0.0.1:{_TEST_PORT}")
    result = client.predict("Hello", "", api_name="/respond")

    assert result[0] == "Hello"
    assert "Response for: Hello" in result[1]
    assert "Embedding recall" in result[2]
    assert "keyword-a" in str(result[3])
    assert "Code sample" in result[4]
