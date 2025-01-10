from pydantic import BaseModel, Field, constr, conlist
from typing import Union, List, Literal

class Metadata_form(BaseModel):
    """Information about dataset."""

    sex_2 : Literal["male", "female", "other/mixed/unspecified"] = Field(description = "The sex of the subject studied",)
    hardware_4 : constr(max_length=40) = Field(
            description="Name of machine/techonology used for processing biosamples.",
            examples = ["Illumina HiSeq 2000",
                        "Affymetrix SNP6.0",
                        "Agilent miRNA microarrays"]
            )
    assay_count_7 : int = Field(description = "Assay count",)
    type_9 : Literal["geo", "ena", "gxa", "biostudies", "gxa-cx", "ega"] = Field(description = 'Any of ["geo", "ena", "gxa", "biostudies", "gxa-cx", "ega"].')
    experimental_design_10 : constr(max_length=70) = Field(description = "Term for the experimental design",)

    releasedate_12 : int = Field(description = "Year the dataset was released")
    sample_count_13 : int = Field(description = "Number of samlpes in the dataset",)
    assay_by_molecule_14 : Literal['rna assay','dna assay', 'protein assay'] = Field(description = "Asssay type",)
    technology_15 : Literal['array assay','sequencing assay'] = Field(description = "Array or sequencing assay",)
    organism_16: constr(max_length=40) = Field(description = "Latin term for subject organism",
            examples = ["homo sapiens", "mus musculus"]
            )


    study_type_18 : Literal['transcription profiling by array', 'rna-seq of coding rna', 'chip-seq', 'rna-seq of non coding rna', 'comparative genomic hybridization by array', 'other', 'methylation profiling by array', 'chip-chip by tiling array', 'methylation profiling by high throughput sequencing', 'unknown experiment type', 'chip-chip by array', 'genotyping by array', 'microrna profiling by array', 'transcription profiling by tiling array', 'rna-seq of coding rna from single cells', 'dna-seq', 'transcription profiling by rt-pcr', 'cell line - high-throughput sequencing', 'proteomic profiling by array', 'animal - high-throughput sequencing', 'transcription profiling by sage', 'high-throughput sequencing', 'atac-seq', 'rnai profiling by array', 'baseline'] = Field(description = "Study type",)
    # 25 first (including "other" already)

    name_19 : constr(max_length=50) = Field(description = "Name of organization/institute/university where the research was conducted",)
    experimental_factors_20 : constr(max_length = 40) = Field(
            description = "factors considered in the experiment",
            examples =['genotype', 'treatment', 'time', 'cell type', 'organism part', 'compound', 'age', 'cell line', 'sex', 'dose']
            )


    type_21 : Literal['bioassay_data_transformation', 'normalization data transformation protocol', 'nucleic_acid_extraction', 'feature_extraction', 'labeling', 'hybridization', 'image_aquisition', 'specified_biomaterial_action', 'grow', 'growth protocol', 'nucleic acid extraction protocol', 'nucleic acid library construction protocol', 'sample treatment protocol', 'labelling protocol', 'array scanning protocol', 'hybridization protocol', 'nucleic acid sequencing protocol', 'sample collection protocol', 'treatment protocol', 'nucleic acid labeling protocol', 'nucleic acid hybridization to array protocol', 'image_acquisition', 'array scanning and feature extraction protocol', 'high throughput sequence alignment protocol', 'pool', 'sequencing', 'scanning', 'dissection protocol', 'normalization', 'extraction', 'conversion protocol', 'compound', 'growth', 'labelling', 'compound_based_treatment', 'feature extraction', 'unknown_protocol_type', 'treatment', 'immunoprecipitation', 'harvest', 'chip', 'data transformation', 'dissect', 'split', 'immunoprecipitate', 'purify', 'fractionate', 'transformation', 'linear_amplification', 'sampling', "other"] = Field(description = "Name of protocol used",)
    # 50 first + other

    no_of_samples_22 : int = Field(description = "number of samples",)
    no_of_samples_23 : int = Field(description = "number of samples",)

class Study_type_metadata_form(BaseModel):
    study_type_18 : Literal['transcription profiling by array', 'rna-seq of coding rna', 'chip-seq', 'rna-seq of non coding rna', 'comparative genomic hybridization by array', 'other', 'methylation profiling by array', 'chip-chip by tiling array', 'methylation profiling by high throughput sequencing', 'unknown experiment type', 'chip-chip by array', 'genotyping by array', 'microrna profiling by array', 'transcription profiling by tiling array', 'rna-seq of coding rna from single cells', 'dna-seq', 'transcription profiling by rt-pcr', 'cell line - high-throughput sequencing', 'proteomic profiling by array', 'animal - high-throughput sequencing', 'transcription profiling by sage', 'high-throughput sequencing', 'atac-seq', 'rnai profiling by array', 'baseline'] = Field(description = "Study type",)
    # 25 first (including "other" already)

class Unconstrained_metadata_form(BaseModel):
    """Information about dataset."""

    sex_2 : Literal["male", "female", "other/mixed/unspecified"] = Field(description = "The sex of the subject studied",)
    hardware_4 : str = Field(
            description="Name of machine/techonology used for processing biosamples.",
            #examples = ["Illumina HiSeq 2000",
            #            "Affymetrix SNP6.0",
            #            "Agilent miRNA microarrays"]
            )
    assay_count_7 : int = Field(description = "Assay count",)
    type_9 : Literal["geo", "ena", "gxa", "biostudies", "gxa-cx", "ega"] = Field(description = 'Any of ["geo", "ena", "gxa", "biostudies", "gxa-cx", "ega"].')
    experimental_design_10 : str = Field(description = "Term for the experimental design",)

    releasedate_12 : int = Field(description = "Year the dataset was released")
    sample_count_13 : int = Field(description = "Number of samlpes in the dataset",)
    assay_by_molecule_14 : Literal['rna assay','dna assay', 'protein assay'] = Field(description = "Asssay type",)
    technology_15 : Literal['array assay','sequencing assay'] = Field(description = "Array or sequencung assay",)
    organism_16: str = Field(description = "Latin term for subject organism",
                             #examples = ["homo sapiens", "mus musculus"]
            )


    study_type_18 : Literal['transcription profiling by array', 'rna-seq of coding rna', 'chip-seq', 'rna-seq of non coding rna', 'comparative genomic hybridization by array', 'other', 'methylation profiling by array', 'chip-chip by tiling array', 'methylation profiling by high throughput sequencing', 'unknown experiment type', 'chip-chip by array', 'genotyping by array', 'microrna profiling by array', 'transcription profiling by tiling array', 'rna-seq of coding rna from single cells', 'dna-seq', 'transcription profiling by rt-pcr', 'cell line - high-throughput sequencing', 'proteomic profiling by array', 'animal - high-throughput sequencing', 'transcription profiling by sage', 'high-throughput sequencing', 'atac-seq', 'rnai profiling by array', 'baseline'] = Field(description = "Study type",)
    # 25 first (including "other" already)

    name_19 : str = Field(description = "Name of organization/institute/university where the research was conducted",)
    experimental_factors_20 : str = Field(
            description = "factors considered in the experiment",
            #examples =['genotype', 'treatment', 'time', 'cell type', 'organism part', 'compound', 'age', 'cell line', 'sex', 'dose']
            )


    type_21 : Literal['bioassay_data_transformation', 'normalization data transformation protocol', 'nucleic_acid_extraction', 'feature_extraction', 'labeling', 'hybridization', 'image_aquisition', 'specified_biomaterial_action', 'grow', 'growth protocol', 'nucleic acid extraction protocol', 'nucleic acid library construction protocol', 'sample treatment protocol', 'labelling protocol', 'array scanning protocol', 'hybridization protocol', 'nucleic acid sequencing protocol', 'sample collection protocol', 'treatment protocol', 'nucleic acid labeling protocol', 'nucleic acid hybridization to array protocol', 'image_acquisition', 'array scanning and feature extraction protocol', 'high throughput sequence alignment protocol', 'pool', 'sequencing', 'scanning', 'dissection protocol', 'normalization', 'extraction', 'conversion protocol', 'compound', 'growth', 'labelling', 'compound_based_treatment', 'feature extraction', 'unknown_protocol_type', 'treatment', 'immunoprecipitation', 'harvest', 'chip', 'data transformation', 'dissect', 'split', 'immunoprecipitate', 'purify', 'fractionate', 'transformation', 'linear_amplification', 'sampling', "other"] = Field(description = "Name of protocol used",)
    # 50 first + other

    no_of_samples_22 : int = Field(description = "number of samples",)
    no_of_samples_23 : int = Field(description = "number of samples",)
