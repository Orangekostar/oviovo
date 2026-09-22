"""Failure-oriented checks for bounded ScanNet acquisition and native export."""

from __future__ import annotations

import importlib
import json
import ssl
import struct
import subprocess
import threading
import urllib.error
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest


def module():
    return importlib.import_module("src.static_ovmap.module_validation.scannet_download")


def layout_file(tmp_path):
    path = tmp_path / "provided.py"
    path.write_text(
        "BASE_URL = 'http://kaldir.vc.cit.tum.de/scannet/'\n"
        "RELEASES = ['v2/scans', 'v1/scans']\n"
        "RELEASES_TASKS = ['v2/tasks', 'v1/tasks']\n"
        "LABEL_MAP_FILES = ['scannetv2-labels.combined.tsv', 'scannet-labels.combined.tsv']\n"
        "raise RuntimeError('must never execute supplied script')\n"
    )
    return path


def test_uses_supplied_release_layout_without_executing_it(tmp_path):
    api = module()
    ssl_factory = ssl._create_default_https_context
    layout = api.DownloaderLayout.from_script(layout_file(tmp_path))
    assert layout.scene_url("scene0000_00", ".sens") == (
        "https://kaldir.vc.cit.tum.de/scannet/v1/scans/scene0000_00/scene0000_00.sens"
    )
    assert "/v2/scans/" in layout.scene_url("scene0000_00", ".aggregation.json")
    assert ssl._create_default_https_context is ssl_factory
    with pytest.raises(ValueError):
        layout.scene_url("../scene0000_00", ".sens")
    with pytest.raises(ValueError):
        layout.scene_url("scene0000_00", "_2d-label.zip")


@pytest.mark.parametrize("count,step,last,end", [(200, 1, 199, 200), (399, 1, 199, 200), (2345, 11, 2189, 2190)])
def test_schedule_preserves_native_negative_step_and_original_ids(count, step, last, end):
    schedule = module().native_schedule(count)
    assert schedule["step"] == step
    assert schedule["frame_ids"] == list(range(0, last + 1, step))
    assert len(schedule["frame_ids"]) == 200
    assert schedule["end"] == end
    with pytest.raises(ValueError, match="200"):
        module().native_schedule(199)


def test_plan_is_provisional_and_selects_distinct_ordered_families(tmp_path):
    api = module()
    train = tmp_path / "scannetv2_train.txt"
    val = tmp_path / "scannetv2_val.txt"
    train.write_text("\n".join(f"scene{i:04d}_00" for i in range(13)) + "\nscene0004_01\n")
    val.write_text("scene0100_00\nscene0101_00\nscene0102_00\n")
    plan = api.prepare_plan(api.DownloaderLayout.from_script(layout_file(tmp_path)), train, val, ["scene0011"])
    assert plan["status"] == "PROVISIONAL_REQUIRES_AVAILABILITY_CHECK"
    assert len(plan["candidates"]) == 14
    assert [r["role"] for r in plan["candidates"]] == ["fit"] * 8 + ["cal"] * 2 + ["select"] * 2 + ["confirm"] * 2
    assert plan["candidates"][0]["scene_id"] == "scene0000_00"
    assert "scene0004_01" not in [r["scene_id"] for r in plan["candidates"]]
    assert "scene0011_00" not in [r["scene_id"] for r in plan["candidates"]]
    assert len({r["family_id"] for r in plan["candidates"]}) == 14
    val.write_text("scene0000_01\nscene0100_00\n")
    with pytest.raises(ValueError, match="physical.*overlap"):
        api.prepare_plan(api.DownloaderLayout.from_script(layout_file(tmp_path)), train, val, [])


def test_download_requires_authorization_before_network_or_mutation(tmp_path):
    api = module()
    output = tmp_path / "absent"
    with pytest.raises(PermissionError, match="authorization"):
        api.download_selected({}, output, authorized=False)
    assert not output.exists()


def test_existing_download_requires_matching_content_identity(tmp_path):
    api = module()
    path = tmp_path / "asset"
    path.write_bytes(b"valid")
    remote = {"url": "https://example.org/asset", "size_bytes": 5, "etag": '"v1"', "last_modified": None}
    api.record_download(path, remote)
    assert api.verified_download(path, remote)
    path.write_bytes(b"wrong")
    assert not api.verified_download(path, remote)
    path.write_bytes(b"valid")
    assert not api.verified_download(path, {**remote, "etag": '"v2"'})
    receipt = json.loads(Path(str(path) + ".download.json").read_text())
    assert len(receipt["sha256"]) == 64


def sensor_fixture(path, count=400):
    import cv2
    matrix = np.eye(4, dtype="<f4")
    matrix[0, 0] = 200
    matrix[1, 1] = 201
    rgb = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", rgb)
    assert ok
    compressed = zlib.compress(np.full((2, 3), 1234, dtype="<u2").tobytes())
    with path.open("wb") as handle:
        handle.write(struct.pack("<IQ", 4, 4) + b"test")
        for value in (matrix, np.eye(4, dtype="<f4"), matrix, np.eye(4, dtype="<f4")):
            handle.write(value.tobytes())
        handle.write(struct.pack("<iiIIIIfQ", 2, 1, 4, 3, 3, 2, 1000., count))
        for index in range(count):
            pose = np.eye(4, dtype="<f4")
            pose[0, 3] = index
            handle.write(pose.tobytes() + struct.pack("<QQQQ", index, index, len(jpeg), len(compressed)))
            handle.write(jpeg.tobytes() + compressed)
    return jpeg.tobytes()


def test_streaming_export_preserves_sensor_values_schedule_and_resume(tmp_path):
    import cv2
    api = importlib.import_module("src.static_ovmap.module_validation.scannet_frames")
    sensor = tmp_path / "scene0000_00.sens"
    jpeg = sensor_fixture(sensor)
    output = tmp_path / "exported"
    schedule = module().native_schedule(400)
    result = api.export_sensor(sensor, output, schedule)
    assert result["frame_count"] == 200
    assert sorted(int(p.stem) for p in (output / "color").glob("*.jpg")) == list(range(0, 400, 2))
    assert (output / "color/2.jpg").read_bytes() == jpeg
    np.testing.assert_array_equal(cv2.imread(str(output / "depth/2.png"), -1), np.full((2, 3), 1234))
    assert np.loadtxt(output / "pose/2.txt")[0, 3] == 2
    assert np.loadtxt(output / "intrinsic/intrinsic_color.txt")[0, 0] == 200
    assert api.export_sensor(sensor, output, schedule) == result
    (output / "depth/2.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match="export.*changed"):
        api.export_sensor(sensor, output, schedule)


def test_truncated_sensor_cannot_publish_completed_export(tmp_path):
    api = importlib.import_module("src.static_ovmap.module_validation.scannet_frames")
    sensor = tmp_path / "scene0000_00.sens"
    sensor_fixture(sensor, 200)
    sensor.write_bytes(sensor.read_bytes()[:-10])
    output = tmp_path / "exported"
    with pytest.raises(ValueError, match="truncated"):
        api.export_sensor(sensor, output, module().native_schedule(200))
    assert not (output / "export_receipt.json").exists()


def test_v1_sensor_v2_metadata_count_difference_requires_same_native_slots(tmp_path):
    api = importlib.import_module("src.static_ovmap.module_validation.scannet_frames")
    sensor = tmp_path / "scene0000_00.sens"
    sensor_fixture(sensor, 400)
    output = tmp_path / "same_schedule"
    receipt = api.export_sensor(sensor, output, module().native_schedule(401))
    assert receipt["sensor_source_frame_count"] == 400
    assert receipt["schedule"]["source_end"] == 401
    assert receipt["frame_count"] == 200
    assert (output / "color/398.jpg").exists()
    assert not (output / "color/399.jpg").exists()
    with pytest.raises(ValueError, match="schedule"):
        api.export_sensor(sensor, tmp_path / "different_schedule", module().native_schedule(399))


def test_explicit_scannet_root_does_not_depend_on_basename(tmp_path):
    from src.static_ovmap.module_validation.assets import (
        AssetRequirement,
        resolve_assets,
    )
    from src.static_ovmap.module_validation.contracts import StudySpec
    root = tmp_path / "authorized-dataset"
    root.mkdir()
    (root / "scannetv2_train.txt").write_text("scene0000_00\n")
    (root / "scannetv2_val.txt").write_text("scene0100_00\n")
    spec = StudySpec.load(Path(__file__).parents[2] / "docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json")
    result = resolve_assets(spec, {"OVIMAP_DATA_ROOTS": str(root)}, requirements=(
        AssetRequirement("scannet_root", "directory", ("scannet200_release",)),
    ))
    assert result.bindings["scannet_root"].path == root


def test_inventory_reuses_exported_rgbd_instead_of_requiring_sensor_again(tmp_path):
    from src.static_ovmap.module_validation.assets import build_scannet_inventory
    (tmp_path / "scannetv2_train.txt").write_text("scene0000_00\n")
    (tmp_path / "scannetv2_val.txt").write_text("scene0100_00\n")
    scene = tmp_path / "scans/scene0000_00"
    scene.mkdir(parents=True)
    for suffix in module().REQUIRED_SUFFIXES:
        if suffix != ".sens":
            (scene / f"scene0000_00{suffix}").write_text("numDepthFrames = 2345\n" if suffix == ".txt" else "x")
    exported = tmp_path / "exported/scene0000_00"
    for folder, filename in (("color", "0.jpg"), ("depth", "0.png"), ("pose", "0.txt"),
                             ("intrinsic", "intrinsic_color.txt"), ("intrinsic", "intrinsic_depth.txt")):
        (exported / folder).mkdir(parents=True, exist_ok=True)
        (exported / folder / filename).write_text("fixture")
    inventory = build_scannet_inventory(tmp_path)
    assert inventory["scenes"][0]["complete"]
    assert inventory["scenes"][0]["native_input_root"] == str(exported)
    assert inventory["scenes"][0]["schedule"]["step"] == 11
    assert inventory["scenes"][0]["missing_scheduled_frame_ids"] == list(range(11, 2190, 11))


@pytest.mark.parametrize("disconnect_once", [False, True])
def test_https_transfer_resumes_verified_partial_and_rejects_changed_binding(tmp_path, monkeypatch, disconnect_once):
    """Exercise real curl Range transfer over a trusted local HTTPS endpoint."""
    api = module()
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost",
                    "-addext", "subjectAltName=IP:127.0.0.1"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    payload = b"sensor-transfer-content" * 10
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", '"stable"')
            self.end_headers()

        def do_GET(self):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
            ranges.append(start)
            self.send_response(206)
            self.send_header("Content-Length", str(len(payload) - start))
            self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
            self.end_headers()
            if disconnect_once and len(ranges) == 1:
                self.wfile.write(payload[start:start + 17])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(payload[start:])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    monkeypatch.setenv("CURL_CA_BUNDLE", str(cert))
    try:
        url = f"https://127.0.0.1:{server.server_port}/asset"
        remote = api.remote_identity(url)
        path = tmp_path / "download"
        Path(str(path) + ".part").write_bytes(payload[:13])
        Path(str(path) + ".part.json").write_text(json.dumps(remote))
        result = api.transfer(url, path, remote)
        assert path.read_bytes() == payload
        assert ranges == ([13, 30] if disconnect_once else [13])
        assert api.transfer(url, path, remote) == result
        assert ranges == ([13, 30] if disconnect_once else [13])
        other = tmp_path / "other"
        Path(str(other) + ".part").write_bytes(payload[:10])
        Path(str(other) + ".part.json").write_text(json.dumps({**remote, "etag": '"old"'}))
        with pytest.raises(ValueError, match="partial download identity changed"):
            api.transfer(url, other, remote)
        assert not other.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_acquisition_uses_same_family_fallback_and_aborts_on_network_outage(tmp_path, monkeypatch):
    api = module()
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("\n".join(f"scene{i:04d}_00" for i in range(12)) + "\nscene0000_01\n")
    val.write_text("scene0100_00\nscene0101_00\n")
    plan = api.prepare_plan(api.DownloaderLayout.from_script(layout_file(tmp_path)), train, val, [])

    def remote(url):
        if "scene0000_00" in url:
            raise urllib.error.HTTPError(url, 404, "missing", {}, None)
        return {"url": url, "size_bytes": 200, "etag": '"fixture"', "last_modified": None}

    def transfer(url, path, _remote=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("numDepthFrames = 200\n" if path.suffix == ".txt" else "unit fixture")

    monkeypatch.setattr(api, "remote_identity", remote)
    monkeypatch.setattr(api, "transfer", transfer)
    root = tmp_path / "raw"
    result = api.download_selected(plan, root, authorized=True)
    assert result["scene_count"] == 14
    locked = json.loads((root / "acquisition_lock.json").read_text())
    assert locked["selected"][0]["scene_id"] == "scene0000_01"
    assert len({row["family_id"] for row in locked["selected"]}) == 14
    assert len(list((root / "scans").glob("*"))) == 14

    def outage(url):
        raise urllib.error.HTTPError(url, 503, "temporarily unavailable", {}, None)

    monkeypatch.setattr(api, "remote_identity", outage)
    with pytest.raises(urllib.error.HTTPError):
        api.download_selected(plan, tmp_path / "outage", authorized=True)
    assert not (tmp_path / "outage/acquisition_lock.json").exists()
