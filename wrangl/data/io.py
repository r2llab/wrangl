"""
Utilities to save/load data to disk.
"""
import bz2
import typing
import tarfile
import ujson as json
from tqdm import auto as tqdm
from torch.utils.data import Dataset


class AutoDataset(list, Dataset):

    @classmethod
    def process_file(cls, file: typing.TextIO):
        """
        Loads a dataset from a single file.
        Assumes that the data is a JSON file with a list of dictionaries.
        """
        return json.load(file)

    @classmethod
    def merge(cls, datasets):
        """Merge multiple datasets"""
        out = cls()
        for d in datasets:
            out.extend(d)
        return out

    @classmethod
    def load_from_disk(cls, path: typing.Union[str, typing.Iterable], verbose=False):
        """
        Loads a dataset from given `path`, where `path` can be either one file path or an iterator over file paths.
        If `path` is a single file, then `cls.process_file` will be applied.
        If `path` is a single file ending in `.bz2`, then bzip2 decompression will be applied, followed by `cls.process_file`.
        If `path` is a single file ending in `.tar.bz2`, then tar decompression will be applied, followed by bzip2 decompression on every file, followed by `cls.process_file`.
        """
        if isinstance(path, str):
            if path.endswith('.tar.bz2'):
                datasets = []
                with tarfile.open(path, 'r:bz2') as tar:
                    iterator = tar.getmembers()
                    if verbose:
                        iterator = tqdm.tqdm(iterator)
                    for member in iterator:
                        file = tar.extractfile(member)
                        datasets.append(cls.process_file(file))
                return cls.merge(datasets)
            elif path.endswith('.bz2'):
                with bz2.open(path, 'rt') as f:
                    return cls.process_file(f)
            else:
                with open(path, 'rt') as f:
                    return cls.process_file(f)
        else:
            iterator = path
            if verbose:
                iterator = tqdm.tqdm(path)
            return cls.merge(cls.process_file(p) for p in iterator)
