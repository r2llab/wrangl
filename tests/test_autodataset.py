import bz2
import json
import tarfile
import unittest
import tempfile
from wrangl.data.io import AutoDataset


class TestAutoDataset(unittest.TestCase):

    def setUp(self):
        self.examples = [
            dict(x='a', y=1),
            dict(x='b', y=2),
            dict(x='c', y=3),
        ]

    def test_process_file(self):
        with tempfile.TemporaryFile('wt+') as f:
            json.dump(self.examples, f)
            f.flush()
            f.seek(0)
            dataset = AutoDataset.process_file(f)
        self.assertListEqual(self.examples, dataset)

    def test_load_json_from_disk(self):
        with tempfile.NamedTemporaryFile('wt+') as f:
            json.dump(self.examples, f)
            f.flush()
            f.seek(0)
            dataset = AutoDataset.load_from_disk(f.name)
        self.assertListEqual(self.examples, dataset)

    def test_load_bz2_from_disk(self):
        with tempfile.NamedTemporaryFile('wb+', suffix='.bz2') as f:
            s = json.dumps(self.examples)
            f.write(bz2.compress(s.encode('utf8')))
            f.flush()
            f.seek(0)
            dataset = AutoDataset.load_from_disk(f.name)
        self.assertListEqual(self.examples, dataset)

    def test_load_tar_bz2_from_disk(self):
        with tempfile.NamedTemporaryFile(suffix='.tar.bz2') as forig:
            with tarfile.open(forig.name, 'w:bz2') as tar:
                with tempfile.NamedTemporaryFile('wt', suffix='.json') as fa:
                    json.dump(self.examples, fa)
                    fa.flush()
                    with tempfile.NamedTemporaryFile('wt', suffix='.json') as fb:
                        json.dump(list(reversed(self.examples)), fb)
                        fb.flush()
                        tar.add(fa.name)
                        tar.add(fb.name)
            tar.close()
            dataset = AutoDataset.load_from_disk(forig.name)
        self.assertListEqual(self.examples + list(reversed(self.examples)), dataset)


if __name__ == '__main__':
    unittest.main()
