"""
Unit tests for the ProductCatalog module.
"""
import pytest

from engine.catalog import ProductCatalog, ProductInfo


class TestProductCatalog:
    def test_get_existing_product(self) -> None:
        product = ProductCatalog.get("switchboard")
        assert product is not None
        assert product.key == "switchboard"
        assert product.display_name == "Electrical Switchboard"
        assert product.category == "Electrical"

    def test_get_unknown_product_returns_none(self) -> None:
        assert ProductCatalog.get("nonexistent") is None

    def test_list_all_returns_all_products(self) -> None:
        all_products = ProductCatalog.list_all()
        assert len(all_products) >= 7
        assert "cube" in all_products
        assert "switchboard" in all_products
        assert "distribution_board" in all_products

    def test_display_name_known(self) -> None:
        assert ProductCatalog.display_name("ceiling_light") == "LED Ceiling Light"

    def test_display_name_unknown_fallback(self) -> None:
        assert ProductCatalog.display_name("unknown_thing") == "Unknown Thing"

    def test_category_known(self) -> None:
        assert ProductCatalog.category("socket") == "Electrical"
        assert ProductCatalog.category("conduit_box") == "Conduit"

    def test_category_unknown(self) -> None:
        assert ProductCatalog.category("missing") == "Unknown"

    def test_dimensions_known(self) -> None:
        assert ProductCatalog.dimensions("switchboard") == (0.30, 0.30, 0.10)

    def test_dimensions_unknown_fallback(self) -> None:
        assert ProductCatalog.dimensions("missing") == (0.20, 0.20, 0.20)

    def test_exists(self) -> None:
        assert ProductCatalog.exists("junction_box") is True
        assert ProductCatalog.exists("not_real") is False

    def test_product_info_immutable(self) -> None:
        product = ProductCatalog.get("cube")
        assert product is not None
        with pytest.raises(AttributeError):
            product.display_name = "Changed"  # type: ignore[misc]