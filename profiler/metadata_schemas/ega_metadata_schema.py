from pydantic import BaseModel, Field, constr, conlist
from typing import Union, List

class Metadata_form(BaseModel):
    """Information about dataset."""
    study_type: constr(max_length=50) = Field(
            description="Type of the study",
            examples=["Exome Sequencing",
                      "Resequencing",
                      "Whole Genome Sequencing",
                      "Transcriptome Analysis"]
            )

    disease: constr(max_length=40) = Field(
            description="Name of the disease, or group of diseases, studied.",
            examples = ["breast cancer",
                        "melanoma",
                        "medulloblastoma",
                        "diabetes"]
            )
    biosamples: int = Field(
            description="Number of biosamples in the dataset used.",)
    technology: constr(max_length=40) = Field(
            description="Name of machine/techonology used for processing biosamples.",
            examples = ["Illumina HiSeq 2000",
                        "Affymetrix SNP6.0",
                        "Agilent miRNA microarrays"]
            )
    date: int = Field(description="year the study/data was released",
                      examples = [2009,2014])
