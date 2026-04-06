"""Data models for the pipeline."""


class Record:
    def __init__(self, record_id: int, data: dict):
        self.record_id = record_id
        self.data = data

    def to_dict(self) -> dict:
        return {"id": self.record_id, "data": self.data}
