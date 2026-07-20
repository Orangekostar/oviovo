"""Persistent JSON-lines subprocess client for detector workers."""

from __future__ import annotations

import json
import select
import subprocess
import threading
import time
from typing import Any


class JsonLineWorkerClient:
    """Send one JSON request per line and read one JSON response per line."""

    def __init__(
        self,
        *,
        command: list[str],
        env: dict[str, str],
        cwd: str,
        request_timeout_sec: float,
        max_response_chars: int | None = None,
    ) -> None:
        self.command = list(command)
        self.env = dict(env)
        self.cwd = str(cwd)
        self.request_timeout_sec = float(request_timeout_sec)
        if max_response_chars is not None and (
            type(max_response_chars) is not int or max_response_chars <= 0
        ):
            raise ValueError("max_response_chars must be a positive integer or None")
        self.max_response_chars = max_response_chars
        self._process: subprocess.Popen[str] | None = None
        self._next_request_id = 1

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.start()
        assert self._process is not None
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Worker process was started without stdin/stdout pipes.")

        request_id = self._next_request_id
        self._next_request_id += 1
        request_payload = dict(payload)
        request_payload["id"] = request_id

        result: dict[str, Any] = {}

        def _exchange() -> None:
            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            try:
                result["response"] = self._exchange_request(self._process, request_id, request_payload)
            except BaseException as exc:
                result["exception"] = exc

        timeout_sec = max(0.001, self.request_timeout_sec)
        thread = threading.Thread(target=_exchange, daemon=True)
        thread.start()
        thread.join(timeout_sec)
        if thread.is_alive():
            self._close_after_error()
            raise TimeoutError(f"JSON-line worker request {request_id} timed out.")
        if "exception" in result:
            self._close_after_error()
            raise result["exception"]
        return result["response"]

    def _exchange_request(
        self,
        process: subprocess.Popen[str],
        request_id: int,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(request_payload, separators=(",", ":")) + "\n")
        process.stdin.flush()

        if process.poll() is not None:
            raise RuntimeError(f"JSON-line worker exited before response for request {request_id}.")
        readable, _, _ = select.select([process.stdout], [], [], max(0.001, self.request_timeout_sec))
        if not readable:
            raise TimeoutError(f"JSON-line worker request {request_id} timed out.")
        if self.max_response_chars is None:
            line = process.stdout.readline()
        else:
            line = process.stdout.readline(self.max_response_chars + 1)
            if len(line) > self.max_response_chars or (
                line != "" and not line.endswith("\n")
            ):
                raise RuntimeError(
                    "JSON-line worker response exceeded "
                    f"{self.max_response_chars} characters before newline "
                    f"for request {request_id}."
                )
        if line == "":
            raise RuntimeError(f"JSON-line worker closed stdout before response for request {request_id}.")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSON-line worker returned malformed JSON for request {request_id}.") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"JSON-line worker returned non-dict JSON for request {request_id}.")
        response_id = response.get("id")
        if type(response_id) is not int or response_id != request_id:
            raise RuntimeError(f"JSON-line worker response id mismatch: expected {request_id}, got {response_id!r}.")
        return response

    def _close_after_error(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            return
        try:
            process.kill()
            process.wait(timeout=2.0)
        except Exception:
            pass

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is not None:
                return
        except Exception:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            return
        try:
            process.terminate()
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            return
        try:
            process.kill()
            process.wait(timeout=2.0)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
