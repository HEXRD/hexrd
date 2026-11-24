"""Wire integrity and bounded acquisition tests for the isolated live sandbox."""

import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest
from scipy.sparse import load_npz

pytest.importorskip('zmq')
pytest.importorskip('lz4.block')
SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stream_fit_grains.py'
spec = importlib.util.spec_from_file_location('stream_fit_grains', SCRIPT)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def header(count=2):
    return [live.json_bytes(dict(htype='dheader-1.0', series=3, header_detail='basic')),
            live.json_bytes(dict(nimages=count, x_pixels_in_detector=4,
                                 y_pixels_in_detector=3, omega_start=5,
                                 omega_increment=.25))]


def image(index=0, dtype='uint32'):
    pixels = np.arange(12, dtype=dtype).reshape(3, 4)
    return live.encode_image(pixels, 3, index, [5+index*.25, 5+(index+1)*.25], .01)


@pytest.mark.parametrize('corrupt', [False, True])
def test_pending_checksums_gate_finalization_and_latch_failure(tmp_path, monkeypatch, corrupt):
    from concurrent.futures import ThreadPoolExecutor, wait
    from threading import Event

    started, release = Event(), Event()
    original = live.verify_image_checksum
    def delayed(parts):
        started.set()
        assert release.wait(5)
        original(parts)
    monkeypatch.setattr(live, 'verify_image_checksum', delayed)
    capture = live.Capture(tmp_path / 'capture', wire_spool=True, fit_threshold=3.,
                           defer_checksums=True)
    try:
        capture.accept(header(1))
        parts = image()
        if corrupt:
            message = json.loads(parts[0]); message['hash'] = 'wrong'
            parts[0] = live.json_bytes(message)
        capture.accept(parts)
        assert started.wait(1)
        assert json.loads(capture.snapshot().read_text())['checksum_status'] == 'pending'
        with ThreadPoolExecutor(1) as pool:
            final = pool.submit(capture.accept, [live.json_bytes(dict(htype='dseries_end-1.0', series=3))])
            try:
                assert wait([final], timeout=.02).not_done
                assert not capture.complete
            finally:
                release.set()
            if corrupt:
                with pytest.raises(ValueError, match='checksum'):
                    final.result()
                assert not capture.complete and capture.checksum_status == 'failed'
                with pytest.raises(ValueError, match='checksum'):
                    capture.snapshot()
                with pytest.raises(ValueError, match='checksum'):
                    capture.accept([live.json_bytes(dict(htype='dseries_end-1.0', series=3))])
            else:
                final.result()
                assert capture.complete
                assert json.loads(capture.snapshot().read_text())['checksum_status'] == 'verified'
    finally:
        release.set()
        capture.close()


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'uint32', 'int32', 'float64'])
@pytest.mark.parametrize('wire', [False, True])
def test_lossless_wire_and_spool(tmp_path, dtype, wire):
    from hexrd.core import imageseries

    capture = live.Capture(tmp_path / 'capture', wire_spool=wire)
    capture.accept(header())
    original = np.arange(12, dtype=dtype).reshape(3, 4)
    for index in range(2):
        capture.accept(image(index, dtype))
    prefix = capture.snapshot()
    old_bytes = prefix.read_bytes()
    capture.accept([live.json_bytes(dict(htype='dseries_end-1.0', series=3))])
    final = capture.snapshot()
    assert prefix.read_bytes() == old_bytes
    assert json.loads(final.read_bytes())['complete']
    live.register_spool_adapter()
    series = imageseries.open(str(final), 'eiger-live-spool')
    for index in range(len(series)):
        pixels = series[index].toarray()
        assert pixels.dtype == original.dtype
        assert pixels.tobytes() == original.tobytes()


@pytest.mark.parametrize('wire', [False, True])
@pytest.mark.parametrize('dtype', ['f4', 'f8', '>f4', '>f8'])
def test_exact_half_count_replay(tmp_path, dtype, wire):
    from hexrd.core import imageseries

    original = np.array([0., .5, 1., 1.5, 25.5, 1000., 1000.5, 2**23+.5,
                         2**24, 2**30, 2**31-128., .5], dtype=dtype).reshape(3, 4)
    before = original.tobytes()
    capture = live.Capture(tmp_path / 'capture', wire_spool=wire)
    capture.accept(header(1))
    parts = live.encode_image(original, 3, 0, [5, 5.25], .01, source_scale=.5)
    capture.accept(parts)
    live.register_spool_adapter()
    series = imageseries.open(str(capture.snapshot()), 'eiger-live-spool')
    assert series[0].toarray().tobytes() == before == original.tobytes()
    if wire:
        with zipfile.ZipFile(capture.root / 'frames/000000.zip') as stored:
            assert [stored.read(str(i)) for i in range(len(parts))] == parts
    for value in (.25, -0., np.inf, 2**31):
        with pytest.raises(ValueError, match='refusing'):
            live.encode_image(np.array([[value]], dtype='f8'), 3, 0, [0, .1], .01,
                              source_scale=.5)


@pytest.mark.parametrize('dtype,scale', [('u2', 1.), ('i4', 1.), ('f4', .5), ('f8', .5)])
@pytest.mark.parametrize('wire', [False, True])
def test_fit_support_preserves_raw_pixels_and_bounds_snapshots(tmp_path, dtype, scale, wire):
    from hexrd.core import imageseries

    original = np.arange(12, dtype=dtype).reshape(3, 4)
    if scale == .5:
        original *= scale
    capture = live.Capture(tmp_path / 'capture', wire_spool=wire, fit_threshold=3.)
    capture.accept(header(2))
    capture.accept(live.encode_image(original, 3, 0, [0., .1], .01, scale))
    prefix = capture.snapshot()
    capture.accept(live.encode_image(original, 3, 1, [.1, .2], .01, scale))
    live.register_spool_adapter()
    raw = imageseries.open(str(prefix), 'eiger-live-spool')
    support = imageseries.open(str(prefix), 'eiger-live-spool', fit_threshold=3.)
    lower = imageseries.open(str(prefix), 'eiger-live-spool', fit_threshold=2.)
    assert len(support) == 1
    with pytest.raises(IndexError):
        support[1]
    assert raw[0].toarray().tobytes() == lower[0].toarray().tobytes() == original.tobytes()
    expected = original.copy()
    expected[(expected >= 0) & (expected <= 3.)] = 0
    assert support[0].toarray().tobytes() == expected.tobytes()
    from hexrd.core.instrument._spotutils import sparse_frame_view
    view = sparse_frame_view(support)
    rows, cols = np.indices(original.shape)
    actual = view.gather(np.array([[0]]), np.array([0, 12]), rows.ravel(), cols.ravel())
    assert actual.tobytes() == expected.ravel().tobytes()
    with (capture.root / 'fit-support.bin').open('r+b') as stored:
        offset = json.loads(prefix.read_text())['fit_records'][0]['arrays'][0][0]
        stored.seek(offset)
        byte = stored.read(1)
        stored.seek(offset)
        stored.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match='checksum'):
        support[0]
    assert raw[0].toarray().tobytes() == original.tobytes()


def test_fit_support_respects_numpy_scalar_threshold_precision(tmp_path):
    from hexrd.core import imageseries

    original = np.full((3, 4), 2**24+2, dtype='f4')
    threshold = float(2**24+1.5)
    capture = live.Capture(tmp_path / 'capture', wire_spool=True, fit_threshold=threshold)
    capture.accept(header(1))
    capture.accept(live.encode_image(original, 3, 0, [0., .1], .01))
    live.register_spool_adapter()
    snapshot = str(capture.snapshot())
    weak = imageseries.open(snapshot, 'eiger-live-spool', fit_threshold=threshold)
    promoted = imageseries.open(snapshot, 'eiger-live-spool', fit_threshold=np.float64(threshold))
    assert not weak[0].toarray().any()
    assert promoted[0].toarray().tobytes() == original.tobytes()


def test_wire_capture_replay_never_opens_scientific_inputs(tmp_path):
    from types import SimpleNamespace

    capture = live.Capture(tmp_path / 'capture', wire_spool=True)
    capture.accept(header(2))
    capture.accept(image(0))
    args = SimpleNamespace(config='must-not-be-opened.yml', wire_input=capture.root,
                           frames=None, rate=100, source_scale=1.)
    with pytest.raises(ValueError, match='incomplete'):
        list(live.replay_messages(args))
    args.frames = 1
    messages = list(live.replay_messages(args))
    assert len(messages) == 3 and json.loads(messages[0][1])['nimages'] == 1
    assert messages[1] == image(0)
    assert json.loads(messages[-1][0]) == dict(htype='dseries_end-1.0', series=3)
    capture.accept(image(1))
    args.frames = None
    assert list(live.replay_messages(args))[1:-1] == [image(0), image(1)]


def test_reusing_raw_archive_preserves_full_pixels_and_rejects_wrong_frames(tmp_path):
    from hexrd.core import imageseries

    original = live.Capture(tmp_path / 'original', wire_spool=True)
    original.accept(header(2))
    original.accept(image(0))
    original.accept(image(1))
    capture = live.Capture(tmp_path / 'capture', fit_threshold=3., raw_archive=original.root)
    capture.accept(header(2))
    capture.accept(image(0))
    live.register_spool_adapter()
    raw = imageseries.open(str(capture.snapshot()), 'eiger-live-spool')
    assert raw[0].toarray().tobytes() == live.decode_image(image(0)).tobytes()
    with pytest.raises(IndexError):
        raw[1]
    assert not list((capture.root / 'frames').iterdir())
    wrong = image(1)
    wrong[4] = live.json_bytes(dict(omega=[6., 6.25], source_dtype='<u4'))
    with pytest.raises(ValueError, match='Raw archive'):
        capture.accept(wrong)
    assert len(capture.omega) == 1


@pytest.mark.parametrize('dtype', ['u4', 'i4', 'f4', 'f8'])
def test_sparse_replay_converts_values_before_image_inflation(dtype):
    from scipy.sparse import csr_array

    dense = np.arange(12, dtype=dtype).reshape(3, 4)
    scale = .5 if dense.dtype.kind == 'f' else 1.
    if scale == .5:
        dense *= scale
    pixels = csr_array(dense)
    before = [a.tobytes() for a in (pixels.data, pixels.indices, pixels.indptr)]
    actual = live.encode_image(pixels, 3, 0, [0., .1], .01, scale)
    expected = live.encode_image(dense, 3, 0, [0., .1], .01, scale)
    assert actual == expected
    assert before == [a.tobytes() for a in (pixels.data, pixels.indices, pixels.indptr)]


def test_saturated_and_buffer_pixels_are_retained(tmp_path):
    original = np.zeros((3, 4), dtype='uint32')
    original[1, 2] = np.iinfo('uint32').max
    capture = live.Capture(tmp_path / 'capture')
    capture.accept(header(1))
    capture.accept(live.encode_image(original, 3, 0, [0, .1], .01))
    np.testing.assert_array_equal(load_npz(capture.root / 'frames/000000.npz').toarray(), original)


@pytest.mark.parametrize('value', [-1, -0., .5, np.nan, np.inf, 2**32])
def test_replay_refuses_lossy_conversion(value):
    pixels = np.array([[value]])
    before = pixels.tobytes()
    with pytest.raises(ValueError, match='refusing'):
        live.encode_image(pixels, 3, 0, [0, .1], .01)
    assert pixels.tobytes() == before


@pytest.mark.parametrize('encoding', ['<', '>', 'lz4<'])
def test_raw_endian_and_lz4(encoding):
    pixels = np.arange(12, dtype=np.dtype('uint16').newbyteorder(encoding[-1])).reshape(3, 4)
    parts = image()
    if encoding == 'lz4<':
        import lz4.block
        payload = lz4.block.compress(pixels, store_size=False)
    else:
        payload = pixels.tobytes()
    parts[0] = live.json_bytes(dict(htype='dimage-1.0', series=3, frame=0))
    parts[1] = live.json_bytes(dict(htype='dimage_d-1.0', shape=[4, 3],
                                   type='uint16', encoding=encoding, size=len(payload)))
    parts[2] = payload
    np.testing.assert_array_equal(live.decode_image(parts), pixels)


def test_corrupt_or_oversized_frame_rejected_before_allocation():
    parts = image()
    parts[2] = bytes([parts[2][0] ^ 1]) + parts[2][1:]
    with pytest.raises(ValueError, match='checksum'):
        live.decode_image(parts)
    parts = image()
    desc = json.loads(parts[1])
    desc['shape'] = [1000000, 1000000]
    parts[1] = live.json_bytes(desc)
    with pytest.raises(ValueError, match='oversized'):
        live.decode_image(parts)


def test_unknown_codec_rejected_with_a_useful_error():
    parts = image()
    description = json.loads(parts[1])
    description['encoding'] = 'unknown<'
    parts[1] = live.json_bytes(description)
    with pytest.raises(ValueError, match='Unsupported Eiger codec: unknown'):
        live.decode_image(parts)


def test_receiver_retains_service_after_provisional_results(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import zmq

    class Worker:
        returncode = 0
        finished = False
        def poll(self):
            return 0 if self.finished else None

    worker, submitted, stopped = Worker(), [], []
    socket = SimpleNamespace(setsockopt=lambda *a: None, connect=lambda *a: None)
    context = SimpleNamespace(socket=lambda *a: nullcontext(socket))
    monkeypatch.setattr(zmq, 'Context', lambda: nullcontext(context))
    monkeypatch.setattr(live, 'start_worker', lambda args: worker)
    monkeypatch.setattr(live, 'stop_worker', stopped.append)

    def messages(*args):
        yield header(5), None
        for index in range(5):
            yield image(index), None
        yield [live.json_bytes(dict(htype='dseries_end-1.0', series=3))], None

    def submit(service, args, snapshot, final=False):
        assert service is worker
        metadata = json.loads(snapshot.read_text())
        submitted.append((metadata['count'], final))
        output = tmp_path / f'result-{metadata["count"]}'
        output.mkdir()
        (output / 'result.json').write_text('{}')
        worker.finished = final
        return output

    monkeypatch.setattr(live, 'prepared_messages', messages)
    monkeypatch.setattr(live, 'submit_analysis', submit)
    args = SimpleNamespace(config='unused.yml', output=tmp_path / 'capture',
                           endpoint='unused', decode_workers=1, timeout=10, checkpoint=2)
    live.receive(args)
    assert submitted == [(2, False), (4, False), (5, True)]
    assert stopped == [worker]
    assert json.loads((args.output / 'status.json').read_text())['complete']


@pytest.mark.parametrize('borrow', [False, True])
def test_reusable_decoder_and_public_fallback_preserve_pixels(borrow):
    buffer = live.DecodeBuffer()
    for index in range(3):
        parts = image(index)
        if borrow:
            parts[2] = memoryview(parts[2])
        expected = live.decode_image(parts)
        actual = live.decode_image(parts, buffer)
        assert actual.tobytes() == expected.tobytes()
    buffer.decode = None
    assert live.decode_image(parts, buffer).tobytes() == expected.tobytes()


@pytest.mark.parametrize('scale', [1., .5])
def test_sparse_wire_capture_avoids_inflation_and_preserves_raw_pixels(tmp_path, monkeypatch, scale):
    import h5py
    import hdf5plugin
    from hexrd.core import imageseries
    from hexrd.core.extensions import transforms as native
    if not hasattr(native, '_EigerSupport'):
        pytest.skip('Native Eiger support unavailable')

    counts = np.resize(np.array([0, 5, 6, 7, 10001, 2**24+1, 2**31+1, 2**32-1], dtype='u4'), (32, 32))
    with h5py.File(tmp_path / 'source.h5', 'w') as source:
        dataset = source.create_dataset('counts', data=counts, chunks=counts.shape,
                                       **hdf5plugin.Bitshuffle(cname='lz4'))
        source.flush()
        _, payload = dataset.id.read_direct_chunk((0, 0))
    parts = image()
    logical = counts if scale == 1. else counts.astype('f8')*scale
    parts[0] = live.json_bytes(dict(htype='dimage-1.0', series=3, frame=0,
                                   hash=live.hashlib.md5(payload).hexdigest()))
    parts[1] = live.json_bytes(dict(htype='dimage_d-1.0', shape=[32, 32],
                                   type='uint32', encoding='bs32-lz4<', size=len(payload)))
    parts[2] = memoryview(payload)
    parts[4] = live.json_bytes(dict(omega=[5., 5.25], source_dtype=logical.dtype.str,
                                   source_scale=scale))
    header_parts = header(1)
    config = json.loads(header_parts[1]);config.update(x_pixels_in_detector=32, y_pixels_in_detector=32)
    header_parts[1] = live.json_bytes(config)
    capture = live.Capture(tmp_path / 'capture', wire_spool=True, fit_threshold=3.)
    try:
        capture.accept(header_parts)
        original = live.decode_pixels
        def forbid(*args):
            raise AssertionError('Sparse wire preparation must not inflate pixels')
        monkeypatch.setattr(live, 'decode_pixels', forbid)
        capture.accept(parts)
        assert capture._decoders.buffer.data is None
        monkeypatch.setattr(live, 'decode_pixels', original)
        capture.accept([live.json_bytes(dict(htype='dseries_end-1.0', series=3))])
        live.register_spool_adapter()
        snapshot = str(capture.snapshot())
        raw = imageseries.open(snapshot, 'eiger-live-spool')
        support = imageseries.open(snapshot, 'eiger-live-spool', fit_threshold=3.)
        assert raw[0].toarray().tobytes() == logical.tobytes()
        expected = logical.copy();expected[expected <= 3.] = 0
        assert support[0].toarray().tobytes() == expected.tobytes()
    finally:
        capture.close()


def test_parallel_preparation_commits_only_completed_ordered_prefix(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    capture = live.Capture(tmp_path / 'capture')
    capture.accept(header(3))
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(capture.prepare_image, image(index)) for index in [2, 1, 0]]
        prepared = [future.result() for future in futures]
    assert not capture.omega  # Preparation alone must never publish acquired rows.
    for index, result in enumerate(reversed(prepared)):
        capture.accept(image(index), result)
    assert len(capture.omega) == 3
    for index in range(3):
        expected = live.decode_image(image(index))
        assert load_npz(capture.root / 'frames' / f'{index:06d}.npz').toarray().tobytes() == expected.tobytes()


def test_bounded_receiver_queue_publishes_every_frame_in_order(tmp_path):
    from collections import deque

    end = [live.json_bytes(dict(htype='dseries_end-1.0', series=3))]
    messages = deque([header(12), *(image(i) for i in range(12)), end])
    class Socket:
        def poll(self, timeout):
            return bool(messages)
        def recv_multipart(self, *, copy=True):
            return messages.popleft()
    capture = live.Capture(tmp_path / 'capture')
    for parts, prepared in live.prepared_messages(Socket(), capture, 3, 1):
        capture.accept(parts, prepared)
    assert capture.complete and len(capture.omega) == 12
    assert len(list((capture.root / 'frames').glob('*.npz'))) == 12


@pytest.mark.parametrize('borrow', [False, True])
def test_dectris_bitshuffle_lz4_payload(tmp_path, borrow):
    import h5py
    import hdf5plugin

    pixels = np.arange(32*32, dtype='uint32').reshape(32, 32)
    with h5py.File(tmp_path / 'chunk.h5', 'w') as handle:
        dataset = handle.create_dataset('counts', data=pixels, chunks=pixels.shape,
                                        **hdf5plugin.Bitshuffle(cname='lz4'))
        handle.flush()
        _, payload = dataset.id.read_direct_chunk((0, 0))
    parts = image()
    parts[0] = live.json_bytes(dict(htype='dimage-1.0', series=3, frame=0))
    parts[1] = live.json_bytes(dict(htype='dimage_d-1.0', shape=[32, 32],
                                   type='uint32', encoding='bs32-lz4<', size=len(payload)))
    parts[2] = memoryview(payload) if borrow else payload
    np.testing.assert_array_equal(live.decode_image(parts), pixels)
    buffer = live.DecodeBuffer()
    np.testing.assert_array_equal(live.decode_image(parts, buffer), pixels)
    buffer.bitshuffle = None
    np.testing.assert_array_equal(live.decode_image(parts, buffer), pixels)
    parts[0] = live.json_bytes(dict(htype='dimage-1.0', series=3, frame=0))
    parts[2] = b'\0'*8 + parts[2][8:]
    with pytest.raises(ValueError, match='bitshuffle frame size'):
        live.decode_image(parts, buffer)


def test_frame_gap_duplicate_and_incomplete_end_are_not_silently_accepted(tmp_path):
    capture = live.Capture(tmp_path / 'capture')
    capture.accept(header())
    with pytest.raises(ValueError, match='Missing, duplicate'):
        capture.accept(image(1))
    capture.accept(image())
    with pytest.raises(ValueError, match='Missing, duplicate'):
        capture.accept(image())
    with pytest.raises(ValueError, match='Incomplete'):
        capture.accept([live.json_bytes(dict(htype='dseries_end-1.0', series=3))])
    assert not capture.complete
    assert len(list((capture.root / 'frames').iterdir())) == 1


def test_basic_hardware_stream_without_sandbox_appendix(tmp_path):
    capture = live.Capture(tmp_path / 'capture')
    capture.accept(header())
    capture.accept(image()[:4])
    assert capture.omega == [[5, 5.25]]
    assert capture.dtype == np.dtype('uint32')


def test_snapshot_reads_only_declared_prefix(tmp_path):
    from hexrd.core import imageseries

    capture = live.Capture(tmp_path / 'capture')
    capture.accept(header())
    capture.accept(image())
    path = capture.snapshot()
    capture.accept(image(1))
    live.register_spool_adapter()
    series = imageseries.open(str(path), 'eiger-live-spool')
    assert len(series) == 1
    with pytest.raises(IndexError):
        series[1]
    assert series.metadata['omega'].tolist() == [[5, 5.25]]


def test_actual_zmq_multipart_transport(tmp_path):
    import zmq

    capture = live.Capture(tmp_path / 'capture')
    with (zmq.Context() as context, context.socket(zmq.PUSH) as sender,
          context.socket(zmq.PULL) as receiver):
        endpoint = 'inproc://eiger-test'
        sender.bind(endpoint)
        receiver.connect(endpoint)
        end = [live.json_bytes(dict(htype='dseries_end-1.0', series=3))]
        for message in [header(1), image(), end]:
            sender.send_multipart(message)
            assert receiver.poll(1000)
            capture.accept(receiver.recv_multipart())
    assert capture.complete


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux sandbox')
def test_ctrl_c_while_waiting_for_stream(tmp_path):
    log = tmp_path / 'receiver.log'
    with log.open('wb') as stream:
        process = subprocess.Popen([sys.executable, str(SCRIPT), 'receive', 'unused.yml',
                                    str(tmp_path / 'capture'), '--endpoint',
                                    f'ipc://{tmp_path}/absent'], stdout=stream, stderr=stream)
    try:
        deadline = time.monotonic() + 15
        while b'Listening' not in log.read_bytes():
            assert process.poll() is None, log.read_text()
            assert time.monotonic() < deadline
            time.sleep(.05)
        start = time.monotonic()
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=3) == 130
        assert time.monotonic()-start < 2
        assert not (tmp_path / 'capture/final/result.json').exists()
    finally:
        live.stop_worker(process)


def test_stale_analysis_can_be_stopped_responsively():
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    start = time.monotonic()
    live.stop_worker(worker)
    assert time.monotonic()-start < 2
    assert worker.poll() is not None


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process groups')
@pytest.mark.parametrize('parent_exits', [False, True])
def test_cancelling_service_also_stops_its_fitting_processes(parent_exits):
    import psutil

    command = (
        "import resource;resource.setrlimit(resource.RLIMIT_AS,(64*1024**2,)*2);"
        "import subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "print(child.pid,flush=True);"
    ) + ('sys.exit(1)' if parent_exits else 'time.sleep(60)')
    worker = subprocess.Popen([sys.executable, '-c', command], stdout=subprocess.PIPE,
                              text=True, start_new_session=True)
    try:
        child = psutil.Process(int(worker.stdout.readline()))
        worker._hexrd_group = True
        if parent_exits:
            assert worker.wait(timeout=2) == 1
        started = time.monotonic()
        live.stop_worker(worker)
        while True:
            try:
                if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            assert time.monotonic() - started < 2
            time.sleep(.01)
    finally:
        live.stop_worker(worker)
        worker.stdout.close()


def test_compare_checks_exact_fits_and_spot_files(tmp_path):
    left, right = [tmp_path / name for name in ('left', 'right')]
    for directory in (left, right):
        directory.mkdir()
        np.savez(directory / 'scientific.npz', fits=np.zeros((2, 15)))
        np.savez(directory / 'grains.npz', grains=np.empty((0, 21)))
        (directory / 'grains.out').write_text('identical fits')
    live.compare(left, right)
    (right / 'grains.out').write_text('changed fits')
    with pytest.raises(AssertionError, match='Output differs'):
        live.compare(left, right)


@pytest.mark.parametrize('archive_failure', [False, True])
def test_fitting_starts_before_archive_finishes_and_writer_errors_block_completion(
        tmp_path, monkeypatch, archive_failure):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import threading
    from hexrd.hedm import find_orientations as fo
    from hexrd.hedm.cli import fit_grains as cli

    output = tmp_path / 'analysis'
    gate, started = threading.Event(), threading.Event()
    result = fo.FindOrientationsResult(np.array([[1., 0., 0., 0.]]).T,
                                      np.ones((4, 2)), np.ones(2))
    session = live.AnalysisSession.__new__(live.AnalysisSession)
    session.config = dict(working_dir=str(tmp_path), image_series=dict(data=[]),
                          fit_grains={}, find_orientations=dict(orientation_maps={}))
    session.discovery_workers, session.setup_seconds = 1, 0.
    session.material, session.cancelled = object(), threading.Event()
    session.experiment = lambda config: SimpleNamespace(
        config=config, analysis_dir=str(output), image_series_list=[range(2)],
        accepted_orientations_file=str(output / 'accepted.dat'),
        scored_orientations_file=str(output / 'scored.npz'))
    monkeypatch.setattr(fo, 'find_orientations', lambda *a, **k: result)

    def archive(*args):
        started.set()
        assert gate.wait(2), 'fitting was blocked on archive publication'
        if archive_failure:
            raise OSError('archive publication failed')
    monkeypatch.setattr(fo, 'write_orientation_archive', archive)

    def fit(experiment, initial):
        assert started.wait(2)
        assert not gate.is_set()
        assert (output / 'grains.out').exists()
        gate.set()
        return [(0, 1., 0., initial[0, 3:15])], [123]
    session.fit = fit
    monkeypatch.setattr(cli, 'write_results', lambda *a: None)
    with ThreadPoolExecutor(1) as session.writer:
        if archive_failure:
            with pytest.raises(OSError, match='archive publication failed'):
                session.run(output)
            assert not (output / 'result.json').exists()
        else:
            session.run(output)
            assert (output / 'result.json').exists()


def test_orientation_archive_cancellation_preserves_previous_file(tmp_path):
    from types import SimpleNamespace
    from hexrd.hedm.find_orientations import write_orientation_archive, FindOrientationsResult
    from hexrd.hedm._progress import _cancellation_check

    path = tmp_path / 'scored.npz'
    path.write_bytes(b'previous complete archive')
    result = FindOrientationsResult(np.zeros((4, 0)), np.ones((4, 8)), np.ones(8))
    def cancel():
        raise live.AnalysisCancelled()
    token = _cancellation_check.set(cancel)
    try:
        with pytest.raises(live.AnalysisCancelled):
            write_orientation_archive(result, SimpleNamespace(scored_orientations_file=path))
    finally:
        _cancellation_check.reset(token)
    assert path.read_bytes() == b'previous complete archive'
    assert not path.with_suffix('.tmp').exists()


@pytest.mark.parametrize('readers', [1, 4, 8])
@pytest.mark.parametrize('allowance', [live.FIT_WORKER_MIB, 2 * live.FIT_WORKER_MIB])
def test_persistent_worker_honors_configured_readers_within_budget(monkeypatch, readers, allowance):
    from types import SimpleNamespace

    settings = SimpleNamespace(read_workers=readers)
    session = SimpleNamespace(base=SimpleNamespace(fit_grains=SimpleNamespace(image_cache=settings)))
    limits = []
    monkeypatch.setattr(live, 'allocation_limit', limits.append)
    monkeypatch.setattr(live, 'AnalysisSession', lambda *args, **kwargs: session)
    monkeypatch.setattr(live, 'fit_session', None, raising=False)
    monkeypatch.setattr(live, 'fit_cancelled', None, raising=False)
    live.init_fit_worker('unused.yml', object(), allowance)
    assert limits == [allowance]
    assert session.read_workers == (readers if allowance == 2 * live.FIT_WORKER_MIB else 1)


def test_large_fch5_preserves_grouped_work_under_pool_budget(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace
    import threading

    calls, sizes = [], []
    class Pool:
        def __init__(self, workers, **kwargs):
            calls.append((workers, kwargs['initargs'][-1]))
        def submit(self, function, config, batch, provisional):
            sizes.append(len(batch))
            future = Future()
            future.set_result((123, [(int(row[0]), 1., 0., row[3:15]) for row in batch]))
            return future
    monkeypatch.setattr(live, 'ProcessPoolExecutor', Pool)
    session = live.AnalysisSession.__new__(live.AnalysisSession)
    session.workers, session.pool_workers, session.pool = 2, 0, None
    session.context, session.filename, session.cancelled = object(), 'unused', threading.Event()
    source = SimpleNamespace(shape=(4362, 4148), dtype=np.dtype('f8'))
    class Series:
        shape, dtype = source.shape, source.dtype
        def __len__(self):
            return 3600
    experiment = SimpleNamespace(config=dict(image_series=dict(format='frame-cache',
        data=[dict(args=dict(style='fch5'))])), image_series_list=[Series()])
    initial = np.zeros((2000, 21)); initial[:, 0] = np.arange(2000)
    results, _ = session.fit(experiment, initial)
    assert [row[0] for row in results] == list(range(2000))
    assert calls == [(1, 2 * live.FIT_WORKER_MIB)]
    assert max(sizes) <= 128 and len(sizes) == 16


def test_compare_handles_fortran_trials_empty_grains_and_signed_zero(tmp_path):
    left, right = [tmp_path / name for name in ('left', 'right')]
    values = np.arange(20.).reshape(4, 5)
    for directory, array in [(left, values), (right, np.asfortranarray(values))]:
        directory.mkdir()
        np.savez(directory / 'scientific.npz', trials=array, fits=np.empty((0, 15)))
        np.savez(directory / 'grains.npz', grains=np.empty((0, 21)))
    live.compare(left, right)
    changed = values.copy()
    changed[0, 0] = -0.
    np.savez(right / 'scientific.npz', trials=changed, fits=np.empty((0, 15)))
    with pytest.raises(AssertionError, match='Scientific output differs'):
        live.compare(left, right)


@pytest.mark.parametrize('arguments', [
    ['publish', 'unused.yml', '--frames', '0'],
    ['receive', 'unused.yml', 'unused', '--checkpoint', '1'],
    ['receive', 'unused.yml', 'unused', '--decode-workers', '0'],
    ['publish', 'unused.yml', '--preload-mb', '513'],
])
def test_invalid_cli_sizes_fail_before_opening_data(arguments):
    with pytest.raises(SystemExit) as error:
        live.main(arguments)
    assert error.value.code == 2


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux allocation guards')
def test_receiver_allocation_guard_rejects_excess_and_allows_analysis_child():
    setup = (
        "import importlib.util,resource,json;"
        f"s=importlib.util.spec_from_file_location('live',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
    )
    child = setup + 'm.allocation_limit(4096);print(resource.getrlimit(resource.RLIMIT_AS))'
    parent = setup + '''
import mmap, subprocess, sys
m.allocation_limit(1536)
try:
    mmap.mmap(-1, 1600*1024**2)
except (OSError, MemoryError):
    pass
else:
    raise AssertionError('Allocation guard did not reject excess memory')
''' + f"\nresult=subprocess.check_output([sys.executable,'-c',{child!r}],text=True)\n" + (
        "assert result.strip() == str((4096*1024**2,)*2), result"
    )
    subprocess.run([sys.executable, '-c', parent], check=True, timeout=10)
def test_unobserved_grain_in_acquired_prefix(monkeypatch, tmp_path):
    from hexrd.hedm import fitgrains

    table = np.arange(21, dtype=float)[None, :]
    params = dict(grains_table=table, plane_data=None, instrument=None, imgser_dict={},
                  tth_tol=[1.], eta_tol=[1.], ome_tol=[1.], npdiv=2, refit=None,
                  threshold=1., eta_ranges=[], ome_period=[], analysis_dirname=str(tmp_path),
                  spots_filename=None, return_pull_spots_data=True)
    monkeypatch.setattr(fitgrains, 'paramMP', params, raising=False)
    steps = fitgrains._fit_grain_steps(0)
    next(steps)
    with pytest.raises(RuntimeError, match='simulated number'):
        steps.send(([], {}))
    params['provisional'] = True
    steps = fitgrains._fit_grain_steps(0)
    next(steps)
    with pytest.warns(RuntimeWarning, match='Not enough valid reflections'):
        with pytest.raises(StopIteration) as finished:
            steps.send(([], {}))
    gid, completeness, chisq, fitted, spots = finished.value.value
    assert gid == 0 and np.isnan(completeness) and np.isinf(chisq)
    np.testing.assert_array_equal(fitted, table[0, 3:15])
    assert spots == ([], {})

