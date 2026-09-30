# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import resource
import signal
import struct
import subprocess
import tempfile
import unittest


def string(value):
    encoded = value.encode()
    return struct.pack('<Q', len(encoded)) + encoded


def fixture(tensor_type=0):
    metadata = {
        'general.architecture': 'gpt-oss',
        'gpt-oss.block_count': 24,
        'gpt-oss.expert_count': 32,
        'gpt-oss.embedding_length': 2880,
        'gpt-oss.feed_forward_length': 2880,
        'gpt-oss.attention.head_count': 64,
        'gpt-oss.attention.head_count_kv': 8,
        'gpt-oss.attention.key_length': 64,
        'gpt-oss.attention.value_length': 64,
        'gpt-oss.expert_used_count': 4,
        'gpt-oss.expert_feed_forward_length': 2880,
    }
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, len(metadata))
    for key, value in metadata.items():
        data += string(key)
        if isinstance(value, str):
            data += struct.pack('<I', 8) + string(value)
        else:
            data += struct.pack('<IQ', 10, value)
    data += string('test.weight') + struct.pack('<IQIQ', 1, 4096, tensor_type, 0)
    data += bytes(-len(data) % 32)
    return data + bytes(4096 * 4)


def limit_output():
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))


class RequantSafety(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='tt-lab-requant-')
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        self.source = self.root / 'model.gguf'
        self.original = fixture()
        self.source.write_bytes(self.original)
        self.output = self.root / 'model.ttq'

    def run_requant(self, output=None, *, success=False, limited=False):
        result = subprocess.run(
            ['_out/tt-lab', 'requant', '-m', str(self.source), '-o', str(output or self.output)],
            capture_output=True, text=True, preexec_fn=limit_output if limited else None)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(list(self.root.glob('*.tmp.*')), [])
        return result.stderr

    def test_same_path(self):
        self.assertIn('aliases', self.run_requant(self.source))

    def test_hardlink(self):
        self.output.hardlink_to(self.source)
        self.assertIn('aliases', self.run_requant())
        self.assertEqual(self.output.read_bytes(), self.original)

    def test_symlink(self):
        self.output.symlink_to(self.source.name)
        self.assertIn('aliases', self.run_requant())
        self.assertTrue(self.output.is_symlink())

    def test_validation_failure(self):
        # F16 is a valid GGUF type but unsupported by requant, after output creation.
        self.original = fixture(tensor_type=1)
        self.source.write_bytes(self.original)
        self.output.write_bytes(b'existing TTQ')
        self.assertIn('unsupported tensor type', self.run_requant())
        self.assertEqual(self.output.read_bytes(), b'existing TTQ')

    def test_write_failure(self):
        self.output.write_bytes(b'existing TTQ')
        self.run_requant(limited=True)
        self.assertEqual(self.output.read_bytes(), b'existing TTQ')

    def test_failed_create(self):
        self.run_requant(limited=True)
        self.assertFalse(self.output.exists())

    def test_rename_failure(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_bytes(b'keep')
        self.assertIn('cannot replace', self.run_requant())
        self.assertEqual(marker.read_bytes(), b'keep')

    def test_create(self):
        self.run_requant(success=True)
        self.assertEqual(self.output.read_bytes()[:8], b'TTLBQNT1')

    def test_replace(self):
        self.output.write_bytes(b'existing TTQ')
        self.run_requant(success=True)
        self.assertEqual(self.output.read_bytes()[:8], b'TTLBQNT1')

    def test_replace_symlink(self):
        target = self.root / 'old.ttq'
        target.write_bytes(b'existing TTQ')
        self.output.symlink_to(target.name)
        self.run_requant(success=True)
        self.assertFalse(self.output.is_symlink())
        self.assertEqual(self.output.read_bytes()[:8], b'TTLBQNT1')
        self.assertEqual(target.read_bytes(), b'existing TTQ')


if __name__ == '__main__':
    unittest.main()
