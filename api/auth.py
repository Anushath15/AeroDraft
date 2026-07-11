from pydantic import BaseModel, Field, validator
from typing import Literal, Optional


class SwitchProductCommand(BaseModel):
    type: Literal["switch_product"]
    value: str = Field(..., pattern=r"^(cube|switchboard|socket|ceiling_light|junction_box|conduit_box|distribution_board)$")
    
    @validator("value")
    def validate_product_exists(cls, v):
        from engine.catalog import ProductCatalog
        if v not in ProductCatalog.products():
            raise ValueError(f"Unknown product: {v}")
        return v


class ToggleDemoCommand(BaseModel):
    type: Literal["toggle_demo"]


class ScreenshotCommand(BaseModel):
    type: Literal["screenshot"]
    format: Optional[Literal["png", "jpg"]] = "png"


class ResetCommand(BaseModel):
    type: Literal["reset"]


Command = SwitchProductCommand | ToggleDemoCommand | ScreenshotCommand | ResetCommand
