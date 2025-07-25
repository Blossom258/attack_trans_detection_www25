from typing import Iterator

from daos.defs import CSVReader

class TraceReader(CSVReader):
    def __init__(self, path: str, fn: str = 'TraceItem.csv'):
        super().__init__(path, fn)

