"""OpenSandboxBackend tests — no server, no SDK calls.

The SDK handle is faked at the surface the backend touches
(commands.run / files.write_files / files.read_bytes), so these
tests pin the mapping to deepagents' protocol types (ExecuteResponse,
FileUploadResponse, FileDownloadResponse) without a live sandbox.
Main-feature coverage only: execute / upload / download / id.
"""

from types import SimpleNamespace

import pytest

from runtime.sandbox.opensandbox_backend import OpenSandboxBackend


class _FakeExecution:
    def __init__(self, texts, exit_code=0, error=None):
        self.result = [SimpleNamespace(text=t) for t in texts]
        self.exit_code = exit_code
        self.error = error


class _FakeCommands:
    def __init__(self):
        self.calls = []
        self.next = _FakeExecution(["out\n", "err\n"], exit_code=0)

    def run(self, command, opts=None):
        self.calls.append((command, opts))
        return self.next


class _FakeFiles:
    def __init__(self):
        self.store = {}

    def write_files(self, entries):
        for entry in entries:
            self.store[entry.path] = entry.data

    def read_bytes(self, path):
        if path not in self.store:
            raise FileNotFoundError(path)
        return self.store[path]


class _FakeSandbox:
    def __init__(self):
        self.commands = _FakeCommands()
        self.files = _FakeFiles()


@pytest.fixture()
def backend():
    return OpenSandboxBackend(_FakeSandbox(), "sbx-test")


def test_id_is_the_sandbox_id(backend):
    assert backend.id == "sbx-test"


def test_execute_joins_result_texts(backend):
    response = backend.execute("echo hi")
    assert response.output == "out\nerr\n"
    assert response.exit_code == 0
    assert response.truncated is False


def test_upload_files_writes_entries(backend):
    responses = backend.upload_files(
        [("/work/a.txt", b"hello"), ("/work/b.txt", b"world")]
    )
    assert backend._sandbox.files.store == {
        "/work/a.txt": b"hello",
        "/work/b.txt": b"world",
    }
    assert [r.path for r in responses] == ["/work/a.txt", "/work/b.txt"]
    assert all(r.error is None for r in responses)


def test_download_files_returns_bytes(backend):
    backend._sandbox.files.store["/work/a.txt"] = b"hello"
    responses = backend.download_files(["/work/a.txt", "/work/missing.txt"])
    assert responses[0].content == b"hello"
    assert responses[0].error is None
    assert responses[1].content is None
    assert responses[1].error is not None
