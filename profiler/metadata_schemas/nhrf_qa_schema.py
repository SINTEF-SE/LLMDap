from pydantic import BaseModel, Field, constr, conlist
from typing import Union, List

class Metadata_form(BaseModel):
    """Information about research."""
    biosample_procedure : constr(max_length=500) = Field(
            description = "The procedure for obtaining biosamples in this research")# q1
    genomic_data : constr(max_length=50) = Field(
            description = "The type of genomic data is addressed in the paper")# q2
    cancer_type : constr(max_length=40) = Field( description = "Name of the cancer(s) adressed") #q3
    #q4
    biological_sample : constr(max_length = 15) = Field(description = "What is the biological sample / tissue type",
                                                        examples = ["cell line", "blood", "saliva"]
                                                        )
    #q5
    biosample_origin : constr(max_length = 18) = Field(description = "Is the data from primary tumors, metastatic sites, or both?",
                                                        examples = ["primary tumors", "metastatic sites", "both"]
                                                        )
    #q6
    sample_nature : constr(max_length = 15) = Field(description = "Is there data from paired samples of normal and cancerous tissues?",
                                                        examples = ["paired samples", "only cancerous"]
                                                        )
    #q7
    genomic_position : constr(max_length = 70) = Field(description = "What are the genomic regions or the genomic coordinates of interest (e.g., chromosome, start and end positions)?")
 
    #q8
    mutatuions_and_biomarkers : constr(max_length = 70) = Field(description = "What are particular mutations or biomarkers mentioned in the paper?")

    #q9
    transpript_and_isoforms : constr(max_length=300) = Field(description  = "What are specific transcript variants or isoforms mentioned in the paper?")

    #q10
    variant_types : constr(max_length = 50) = Field(description = "What are specific types of variants (e.g., SNPs, indels, copy number variations, structural variants)?")
 
    #q11
    reference_genome : constr(max_length = 70) = Field(description = "Is the data aligned to a specific reference genome?",
                                                       examples = ["None", "hg19", "hg38", "The zebrafish referencegenome"])
 
    #q12
    mutation : constr(max_length = 200) = Field(description = "The frequency of specific mutation, if any, across cancer types mentioned")
 
    #q13
    demographic : constr(max_length = 80) = Field(description = "Are there specific patient demographics (age, gender, ethnicity) mentioned?")
 
    # Q14
    dataportal : constr(max_length = 80) = Field(description = "Does the scientific paper contain or/and analyse data deposited in private databases or/and open genomic repositories?",
                                                 examples = ["no",
                                                             "GDC (Genomic Data Commons) Data Portal",
                                                             "ArrayExpress",
                                                             "GEO (Gene Expression Omnibus)",
                                                             "EGA (European Genome-phenome Archive)",
                                                             "cBioPortal",
                                                             ]
                                                 )
