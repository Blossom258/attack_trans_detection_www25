from typing import Iterator

from daos.defs import CSVReader
from utils import split_camel_case


class TraceReader(CSVReader):
    def __init__(self, path: str, fn: str = 'TraceItem.csv', signature2keyword: dict = None):
        super().__init__(path, fn)
        self.signature2keyword = signature2keyword
        assert self.signature2keyword is not None

    def iter_read(self) -> Iterator[dict]:
        for row in super().iter_read():
            func_name = ''
            if row['input'] != '0x':
                func_name = self.signature2keyword.get(row['input'][:2 + 8], '')
            func_name = [token.lower() for token in split_camel_case(func_name)]
            yield {'func_name': ' '.join(func_name), **row}
