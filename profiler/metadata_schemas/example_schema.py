# this file shows and explains how a schema for the pipeline should be

from pydantic import BaseModel, Field, constr, conlist
from typing import Union, List, Literal

class Metadata_form(BaseModel):

    # Literal fields contains a full set of possible answers
    fieldname_1 : Literal["yes", "no", "maybe"] = Field(description = "description of the field. This is used for regular retrieval, and fed to the llm to explain what it should answer")

    # constr fields have a maximum number of characters it can generate
    fieldname_2 : constr(max_length=40) = Field(description = "", 
            examples = [
                "answer 1", # example asnwers are shown to the llm.
                "answer 2", # note that the llm can easily be tempted to pick one of these even though the answer is something else
                "answer 3",
                )

    # int fields can only generate integers.
    fieldname_3 : int = Field(description = "")

    # NOTE the llm is also given the field name, so this should describe the field (or could be something generic like "fieldname_2", just dont copy a field from somewhere else and forget to update this)
