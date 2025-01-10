from pydantic import BaseModel, Field, constr, conlist
from typing import Union, List, Literal

class Metadata_form(BaseModel):
    """Information about dataset."""

    dataset_design: constr(max_length = 50) = Field(
                description = 'Dataset design type, related to the questions being addressed by the study, e.g. "time series design", "stimulus or stress design", "genetic modification design"',
                examples = ["Treatment vs control", "time series"],
                )
    organism: constr(max_length = 50) = Field(
                description = 'Species of the samples. Can use common name (e.g. "mouse") or binomial nomenclature/Latin names (e.g. "Mus musculus"). Has EFO expansion.',
                examples = ["Human", "mouse"],
                )
    sample_type:constr(max_length = 50) = Field(
                description = "The type of analyzed sample (e.g. Biosample, FFPE tissue, cell lines, the specific name of the cell lines (HCT116, CaCo2, SK-Mel-24 ...), metastatic tissue)",
                )
    sequencing_technology:constr(max_length = 50) = Field(
                description = "The technology used to obtain genomic/transcriptomic information.",
                examples = ["RNA sequencing","RNAseq", "Whole Exome sequencing", "WES", "exome sequencing", "Whole Genome Sequencing", "WGS"],
                )
    phenotypic_state:constr(max_length = 50) = Field(
                description = "The type of condition(s) the samples/patients showcase and its studied. E.g a phenotypic state could correspond to a specific disease or treatment or condition.",
                examples = ["Metastatic melanoma", "cancer", "normal", "stress", "treatment",],
                )
    tissue:constr(max_length = 50) = Field(
                description = "The type of tissue that the sample comes from or the cell line is derived",
                examples = ["Melanoma", "matching germline tissue"],
                )
    N_sample_conditions:int = Field(
                description = "The number of different conditions in the experiment. E.g. condition 1: treatment with drug A and condition 2: treatment with drug B",
                )
    dataset_size: int = Field(
                description = "Patients, individuals, experimental replicates...",
                )
    experimental_factor: constr(max_length = 50) = Field(
                description = 'Experimental factor (also called experimental variable), the name of the main variable under study in an experiment. E.g. if the factor is "sex" in a human study, the researchers would be comparing between male and female samples, and "sex" is not merely an attribute the samples happen to have. Has EFO (Experimental Factor Ontology) expansion.',
                examples = ["cell type", "Histological Subtype", "sex"],
                )
    experimental_factor_value: constr(max_length = 50) = Field(
                description = 'The value of an experimental factor. E.g. The values for "genotype" factor can be "wild type genotype", "p53-/-". Has EFO expansion.',
                examples = ["HeLa"],
                )
    raw: constr(max_length = 50) = Field(
                description = "Experiment has raw data available.",
                examples = ["yes, private" "yes, public", "no"],
                )
    processed: constr(max_length = 50) = Field(
                description = "Experiment has processed data available.",
                examples = ["yes, private" "yes, public", "no"],
                )
