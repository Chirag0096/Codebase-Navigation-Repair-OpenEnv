"""Shared models for the order system."""


class OrderItem:
    def __init__(self, sku: str, qty: int):
        self.sku = sku
        self.qty = qty
    
    def to_dict(self) -> dict:
        return {"sku": self.sku, "qty": self.qty}
