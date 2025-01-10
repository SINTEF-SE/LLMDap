from pydantic import BaseModel, Field, constr, conlist
from typing import Union, List

class Metadata_form(BaseModel):
    """Information about research."""
    melanoma : constr(max_length=500) = Field(
            description = "Is this paper focusing on melanoma? Does the paper contain genomic data for melanoma?")# q1
    genomic_data: constr(max_length=500) = Field(
            description = "Does this paper contain genomic data?  Is there raw data available, public or private?")# q2
    samples : constr(max_length=500) = Field(
            description = "Does the dataset contain primary and metastatic melanoma samples ? Are there available normal (germ-line) samples?")# q3
    whole_exome_sequencing : constr(max_length=500) = Field(
            description = "Does the dataset contain whole-exome sequencing data from melanomas?")# q4
    transcriptomic_data : constr(max_length=500) = Field(
            description = "Does the dataset contain transcriptomic data from melanomas?")# q5
    experimental_conditions : constr(max_length=500) = Field(
            description = "What are the experimental conditions of the dataset described here?")# q5-2
    hist_subtype : constr(max_length=500) = Field(
            description = "What are the different histological subtypes used in the dataset?")# q6 
    cohort_size : constr(max_length=500) = Field(
            description = "What is the size of the cohort?")# q7 
    q8 : constr(max_length=500) = Field(
            description = "What type of biosamples were used for the dataset (dataset design)? ")# q8 
