import os
import ray
import bz2
import tempfile
import unittest
import ujson as json
from wrangl.data import Fileloader, Processor


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return json.loads(raw)


def create_file(num_lines, start=0):
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl.bz2')
    f.close()
    data = [dict(value=start + i) for i in range(num_lines)]
    with bz2.open(f.name, 'wt') as fzip:
        for row in data:
            fzip.write(json.dumps(row) + '\n')
    return f.name, data


class TestLoadJsonlFiles(unittest.TestCase):

    def setUp(self):
        ray.init()
        self.fnames, self.data = [], []
        for i in range(3):
            fname_i, data_i = create_file(4, start=i*100)
            self.fnames.append(fname_i)
            self.data.extend(data_i)

    def tearDown(self):
        for f in self.fnames:
            os.remove(f)
        ray.shutdown()

    def test_fileloader(self):
        pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
        loader = Fileloader(self.fnames, pool, cache_size=5)
        output = []
        for batch in loader.batch(2):
            output.extend(batch)
        self.assertListEqual(self.data, output)


if __name__ == '__main__':
    unittest.main()