from typing import Optional
from pydantic import BaseModel
from fastapi import Form, File, UploadFile
from enum import Enum

class LLMDapOptions(BaseModel):
    path: Optional[str] = None
    url: Optional[str] = None
    similarity_k: Optional[int] = None
    field_info_to_compare: Optional[str] = None
    schema: Optional[str] = None

class LLMDapOptionsForm(LLMDapOptions):
    @classmethod
    def as_form(
        cls,
        path: Optional[str] = Form(None),
        url: Optional[str] = Form(None),
        similarity_k: Optional[int] = Form(None),
        field_info_to_compare: Optional[str] = Form(None),
        schema: Optional[UploadFile] = File(None),
    ):
        return cls(
            path=path,
            url=url,
            similarity_k=similarity_k,
            field_info_to_compare=field,
            schema=schema.filename if schema else None
        )

# Enum to represent available apps (profilers)
class AppName(str, Enum):
    App1 = "Ydata"
    App2 = "Abstat"
    App3 = "LLMDap"
