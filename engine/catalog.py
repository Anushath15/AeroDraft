"""
Product catalog for AeroDraft MSME demonstrations.
Maps object keys to business-friendly metadata.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass(frozen=True)
class ProductInfo:
    """Business metadata for a renderable object."""
    key: str
    display_name: str
    category: str
    dimensions: Tuple[float, float, float]
    description: str


class ProductCatalog:
    """Central registry of all products available for demonstration."""

    _PRODUCTS: Dict[str, ProductInfo] = {
        "cube": ProductInfo(
            key="cube",
            display_name="Wireframe Cube",
            category="Demo",
            dimensions=(0.20, 0.20, 0.20),
            description="Basic demo object for visualization",
        ),
        "switchboard": ProductInfo(
            key="switchboard",
            display_name="Electrical Switchboard",
            category="Electrical",
            dimensions=(0.30, 0.30, 0.10),
            description="Main electrical distribution panel for residential use",
        ),
        "socket": ProductInfo(
            key="socket",
            display_name="Wall Socket",
            category="Electrical",
            dimensions=(0.08, 0.08, 0.05),
            description="Standard 16A wall power outlet",
        ),
        "ceiling_light": ProductInfo(
            key="ceiling_light",
            display_name="LED Ceiling Light",
            category="Lighting",
            dimensions=(0.25, 0.25, 0.05),
            description="Surface-mounted LED panel for indoor lighting",
        ),
        "junction_box": ProductInfo(
            key="junction_box",
            display_name="Junction Box",
            category="Electrical",
            dimensions=(0.10, 0.10, 0.05),
            description="Electrical wiring junction enclosure",
        ),
        "distribution_board": ProductInfo(
            key="distribution_board",
            display_name="Distribution Board",
            category="Electrical",
            dimensions=(0.40, 0.50, 0.15),
            description="Main power distribution unit for commercial buildings",
        ),
        "conduit_box": ProductInfo(
            key="conduit_box",
            display_name="PVC Conduit Box",
            category="Conduit",
            dimensions=(0.15, 0.15, 0.10),
            description="PVC conduit junction box for cable management",
        ),
    }

    @classmethod
    def get(cls, key: str) -> Optional[ProductInfo]:
        return cls._PRODUCTS.get(key)

    @classmethod
    def list_all(cls) -> Dict[str, ProductInfo]:
        return cls._PRODUCTS.copy()

    @classmethod
    def display_name(cls, key: str) -> str:
        product = cls._PRODUCTS.get(key)
        if product:
            return product.display_name
        return key.replace("_", " ").title()

    @classmethod
    def category(cls, key: str) -> str:
        product = cls._PRODUCTS.get(key)
        return product.category if product else "Unknown"

    @classmethod
    def dimensions(cls, key: str) -> Tuple[float, float, float]:
        product = cls._PRODUCTS.get(key)
        if product:
            return product.dimensions
        return (0.20, 0.20, 0.20)

    @classmethod
    def exists(cls, key: str) -> bool:
        return key in cls._PRODUCTS