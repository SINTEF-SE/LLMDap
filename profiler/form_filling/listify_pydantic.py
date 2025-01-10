from pydantic import BaseModel, Field, conlist, create_model, constr
import pydantic
from typing import Type
import typing

def conlistify_pydantic_model(original_class: Type[BaseModel], min_length=1) -> Type[BaseModel]:
    """ given a pydantic model, make a new one where all the fields have list type of the original fields type.
    Conlist is used to ensure at least min_length entries in each field.

    Example: original_class has field_1 with type int.
    New_class will have field_1 of type conlist(int, min_length).
    New fields have same examples and description as original.
    """


    fields = original_class.__fields__
    new_fields = {}
    for name, field in fields.items():
        
        original_type = original_class.__fields__[name].annotation
        metadata = original_class.__fields__[name].metadata
        if metadata:
            assert len(metadata)==1
            assert original_type == str # only string constraints i implemented for now
            assert type(metadata[0]) == pydantic.types.StringConstraints
            original_type = constr(**metadata[0].__dict__)

        new_type = conlist(original_type, min_length=min_length)
        field = Field(description = original_class.__fields__[name].description,
                      examples = original_class.__fields__[name].examples)
        new_fields[name] = (new_type, field)
    new_class = create_model(f'Listed_{original_class.__name__}', **new_fields)
    new_class.__doc__ = original_class.__doc__
    return new_class


if __name__ == "__main__":
    # test this:

    class Single_Metadata_form(BaseModel):
        """Information about dataset."""
        study_type: constr(max_length=50) = Field(description="Type of the study")
        disease: constr(max_length=40) = Field(description="Name of the disease, or group of diseases, studied.")
        biosamples: int = Field(description="Number of biosamples in the dataset used.")
        technology: constr(max_length=40) = Field(description="Name of machine/techonology used for processing biosamples.",
                                                  examples = ["a","b"])
        date: int = Field(description="year the study/data was released")
    
    class Listed_Metadata_form(BaseModel):
        """Information about dataset."""
        study_type: conlist(constr(max_length=50), min_length=1) = Field(description="Type of the study")
        disease: conlist(constr(max_length=40), min_length=1)= Field(description="Name of the disease, or group of diseases, studied.")
        biosamples: conlist(int, min_length=1) = Field(description="Number of biosamples in the dataset used.")
        technology: conlist(constr(max_length=40),min_length=1) = Field(description="Name of machine/techonology used for processing biosamples.",
                                                                        examples = ["a","b"])
        date: conlist(int, min_length=1) = Field(description="year the study/data was released")
    
    ListedClass = conlistify_pydantic_model(Single_Metadata_form)
    l1 = Listed_Metadata_form.schema()
    l2 = ListedClass.schema()
    del l1["title"]
    del l2["title"]
    assert l1==l2
    print("test passed")
