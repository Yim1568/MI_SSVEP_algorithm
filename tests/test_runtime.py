import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from runtime import arguments


class RuntimeTests(unittest.TestCase):
    def test_duplicates_and_lossless_duration_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache/'YDL_preprocessed.npz').touch()
            argv = ['run','--cache-dir',tmp,'--out-dir',str(cache/'out'),
                    '--subjects','YDL','YDL','--starts','0',
                    '--durations','0.5000001','0.5000002']
            with patch('sys.argv',argv):
                args = arguments('reduced')
            self.assertEqual(args.subjects,['YDL'])
            durations = args.resolved_windows[0]['durations']
            self.assertEqual(len(set(str(d) for d in durations)),2)
            self.assertTrue((args.out_dir/'run_manifest.json').exists())
            with patch('sys.argv',argv), self.assertRaises(FileExistsError):
                arguments('reduced')
