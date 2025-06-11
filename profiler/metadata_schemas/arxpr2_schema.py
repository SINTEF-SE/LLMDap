from pydantic import BaseModel, Field, create_model
from typing import Union, List, Literal
import random


values = dict(
    hardware_4 = dict(
    description="Name of machine/techonology used for processing biosamples.",
        _25= ['illumina hiseq 2000', 'illumina hiseq 2500', 'illumina genome analyzer ii', 'nextseq 500', 'illumina genome analyzer iix', 'illumina hiseq 4000', 'illumina novaseq 6000', 'illumina miseq', 'affymetrix genechip scanner 3000 7g', 'ab solid 4 system', 'illumina hiseq 1500', 'illumina genome analyzer', 'illumina hiseq 1000', 'scanning hardware', 'genepix 4000b [axon instruments]', 'illumina hiseq 3000', 'axon genepix 4000b scanning hardware', 'nextseq 550', 'axon- genepix4000b', 'spectronic helios alpha uv-vis spectrophotometer', 'illumina hiscansq', '418 [affymetrix]', '454 gs flx+', 'hiseq x ten', 'agilent high resolution c scanner'] ,
        v3= ['illumina hiseq 2000', 'illumina hiseq 2500', 'illumina genome analyzer ii', 'illumina genome analyzer iix', 'illumina hiseq 4000', 'nextseq 500', 'illumina novaseq 6000', 'illumina miseq', 'illumina hiseq 1000', 'illumina genome analyzer', 'affymetrix genechip scanner 3000 7g', 'illumina hiseq 1500', 'ab solid 4 system', 'axon- genepix4000b', 'axon genepix 4000b scanning hardware', '418 [affymetrix]', 'scanning hardware', 'illumina hiscansq', 'genepix 4000b [axon instruments]', 'g2565ba dna microarray scanner [agilent]', 'genepix 4000a [axon instruments]', 'ab solid system 3.0', 'illumina hiseq 3000', 'nextseq 550', 'genepix personal 4100a [axon instruments]'],
        _50= ['ab solid system 3.0', 'g2565ba dna microarray scanner [agilent]', 'genepix 4000a [axon instruments]', 'ion torrent proton', 'scanarray express ht', 'affymetrix whole transcript', 'illumina hiseq', '454 gs flx', 'ion torrent pgm', 'genepix personal 4100a [axon instruments]', 'do not apply', 'scanarray 4000xl [perkinelmer]', 'scanarray express ht [perkinelmer]', 'bioanalyser', 'solexa 1g genome analyzer', '-', 'maxwell 16 instrument', 'na', 'affymetrix genechip(r) ht scanner', 'nanodrop nd-1000 spectrophotometer (nanodrop technologies)', 'sony sh800', 'illumina iscan', 'agilent 2100 bioanalyzer', 'ultracentrifuge', 'agilent g2565aa and g2565ba microarray scanner'] ,
        _100= ['chemostat', 'none', 'usegalaxy.org', 'illumina hiseq 2000 platform', 'scanarray [perkinelmer]', 'ab solid system', 'arrayworx biochip reader [applied precision]', 'genepix', 'dasgip bioreactors', 'trizol (invitrogen); fastprep (mp biochemicals)', 'ribozero', 'dna microarray scanner ba [agilent technologies]', '428 [affymetrix]', 'the agilent scanner g2565ba', 'vortex', 'axon instruments genepix 4000b scanner', 'genechip scanner 3000 [affymetrix]', 'nanodrop n1000 spectrophotometer', 'ab 5500xl genetic analyzer', 'genechip scanner 3000 7g', 'genepix 4000b', 'unspecified', 'agilent microarray scanner', 'ion torrent s5 xl', 'scanarray lite [perkinelmer]', 'flat panel airlift-loop reactor with a 1.7 l working volume', 'maxwell® 16 lev simplyrna tissue kit (promega)', 'jencons millennium co2 incubator', 'agilent surescan high resolution dna microarray scanner', 'agilent g2565ba', '10x machine', 'g2565aa dna microarray scanner [agilent]', 'scanarray 6000 [perkin elmer]', 'high resolution c. scanner', 'dell poweredge r820', 'affymetrix genechip command console software', 'illumina nextseq 500', 'agilent gene array scanner g2500a (affymetrix)', 'genechip hybridization oven 640', 'beadarray reader', 'other', 'zeiss palm microbeam system', 'g2565ba dna microarray scanner', 'agilent g2505b scanner', 'covaris dna sonicator', 'packard scanarray4000', 'fluidigm c1', 'bd facsaria iii', 'agilent bravo ngs workstation, microlab star automated liquid handler (hamilton robotics), pcr machine (mj research peltier thermal cycler)', 'bd influx sorter'] ,
        _200= ['agilent�s microarray hybridization oven', 'microcentrifuge', 'axon genepix 4000a scanning hardware', 'illumina hiscan', '454 gs 20', 'affymetrix', '454 sequencing gs flx titanium', 'n/a', 'anaerobic workstation with airlock', 'chromium 10x', 'gene titan', 'scanarray scanning hardware', 'variable atmosphere incubator', 'flexercell fx-4000c', 'affymetrix clariom™ s (400 format)', 'genechip® pico kit (affymetrix)', 'affymetrix gcs 3000', 'applied biosystems 1700 chemiluminescent microarray analyzer', 'ma900, sony', '10x chromium controller', '10x genomics chromium', '10x genomics chromium single cell kit', 'illumina beadarray reader', 'viia 7 real-time pcr system (life technologies)', '454 gs', 'genechip 3000 7g', 'illumina genome analyzer iix;illumina hiseq 2000', 'illumina mousewg-6 v2 expression beadchip', 'affymetrix hg-u133 plus 2.0, fluidics sstation 450', 'gcs3000', 'agilent dna microarray scanner g2565ca', 'sequel', 'agilentm-^rs microarray hybridization oven', 'retsch mm400', '454 gs flx titanium, 454 gs junior', 'epmotion 5075 automated pipetting system (eppendorf, hamburg, germany)', 'qiacube (qiagen)', 'g2505a dna microarray scanner [agilent]', 'proscanarray microarray scanner [perkinelmer]', 'spectrum™ plant total rna kit (sigma aldrich), bioanalyser system (agilent technologies)', 'low input quick amp labeling kit (agilent technologies), nd-1000 spectrophotometer (nanodrop technologies), bioanalyser system (agilent technologies)', 'gene expression hybridisation kit (agilent technologies)', 'agilent technologies scanner g2505c', 'dna microarray scanner (model g2505c, agilent technologies)', 'agilent technologies g2505b scanner', 'g2505b dna microarray scanner', 'tecan ls 300 scanner', 'becton dickinson influx', 'x86 64-apple-darwin13.4.0 (64-bit)', 'hpc services from the university of granada (http://alhambra.ugr.es)', 'sonifier', 'macs', 'illumina genome analyzier iix', 'illumina truseq® stranded mrna library preparation kit (illumina)', 'nanodrop', 'promega maxwellâ® 16 as2000', 'ab solid', 'trizol reagent', 'solexa 2g genome analyzer', 'moflo xdp (beckmancoulter)', 'iridis 4 high performance computing cluster', 'kodak edas 290', 'illumina protocol', 'abi solid v4', 'agilent microarray scanner g2205b', 'agilent technologies microarray scanner', 'genepix 4000b scanner (axon, foster city, ca)', 'ilumina hiscan h166', 'macs (automacs, miltenyi biotec)', 'agilent dna microarray scanner', 'illumina bead array reader', 'agilent scanner g2505b', 'affymetrix hybridization station', 'affymetrix array scanner', 'labfors 3 chemostat (infors, switzerland)', 'dna microarray scanner c (#g2505c; agilent technologies)', 'rneasy plant mini kit (qiagen)', 'affymetrix genechip scanner 3000', 'ab solid 5500', 'maxwell 16 instrument (cat. #as2000, promega, madison, wi)', 'wild m8 zoom stereomicroscope (wild heerbrugg, switzerland)', 'genechip hybridization oven-645 (affymetrix); geneechip fluidics station-450 (affymetrix)', 'affymetrix gene-chip scanner-3000-7g', 'model-g2505b us23502366', 'g2565ba dna scanner', 'truseq stranded mrna sample preparation kit (illumina)', 'qiagen', 'genechip2 hybridization oven 640', 'genechip2 scanner 3000 7g', 'affymetrix genechip instrument system', 'the genechip instrument system', 'agilent g2565ca microarray scanner system', 'agilent g2505c scanner', 'fluidic station  450  and affymetrix scanner 3000', 'illumina genome analyzer ii; illumina genome analyzer iix', 'mousewg-6 v2 0 r1 11278593 a', 'geniom', 'geniom rt analyzer', 'conviron controlled environment chamber', 'illumina genome analyser ii'] ,
    ),
    organism_part_5 = dict(
        description = "Specifies the anatomical source of tissue (e.g. skin)",
        _25= ['liver', 'blood', 'brain', 'lung', 'bone marrow', 'leaf', 'kidney', 'spleen', 'whole organism', 'heart', 'skin', 'root', 'testis', 'colon', 'peripheral blood', 'breast', 'hippocampus', 'pancreas', 'ovary', 'whole blood', 'cerebellum', 'mammary gland', 'thymus', 'whole embryo', 'embryo'] ,
        v3= ['liver', 'blood', 'whole organism', 'leaf', 'bone marrow', 'lung', 'brain', 'skin', 'peripheral blood', 'kidney', 'spleen', 'breast', 'colon', 'heart', 'whole blood', 'mammary gland', 'root', 'hippocampus', 'pancreas', 'testis', 'retina', 'embryo', 'prostate', 'whole embryo', 'spinal cord'],

        _50= ['skeletal muscle', 'shoot', 'retina', 'prostate', 'spinal cord', 'muscle', 'placenta', 'head', 'lymph node', 'inner cell mass', 'small intestine', 'whole body', 'cerebral cortex', 'esophagus', 'ileum', 'hypothalamus', 'seed', 'adipose tissue', 'fruit', 'stomach', 'flower', 'leaves', 'endosperm', 'inflorescence', 'plant embryo'] ,
        _100= ['uterus', '... 5 other values', 'endometrium', 'tumor', 'cortex', 'frontal cortex', 'testes', 'adrenal gland', 'ovaries', 'bone', 'jejunum', 'fetal liver', 'dermis', 'epidermis', 'aerial part', 'striatum', 'duodenum', 'whole brain', 'cervix', '... 2 other values', 'adipose', 'flower bud', 'intestine', 'rectum', 'umbilical cord blood', 'epithelium of small intestine', 'pancreatic islet', 'tonsil', 'bladder', 'leaf blade', 'seedling', 'aorta', 'mycelium', 'prostate gland', 'rosette leaf', 'eye', 'trachea', 'whole animal', 'sperm', 'stem', 'surface ectoderm', 'lens placode', 'thyroid', 'brown adipose tissue', '... 1 other values', 'foreskin', 'salivary gland', '... 4 other values', 'cord blood', 'midbrain'] ,
        _200= ['subcutaneous adipose tissue', 'prefrontal cortex', 'pollen', 'roots', 'large intestine', 'mesenteric lymph node', 'lens', 'glioma', 'rosette', 'gastrocnemius muscle', 'whole worm', 'gut', 'thyroid gland', 'tongue', 'white adipose tissue', 'sigmoid colon', 'distal colon', 'pbmc', 'left ventricle', 'not available', 'wing imaginal disc', 'skeletal muscle tissue', 'whole heart', 'flag leaf', 'islet of langerhans', 'hepatocellular carcinoma', 'cerebrum', 'mycelia', 'prostate cancer', 'breast cancer tumor', 'gill', 'anther', 'humerus', 'root tip', 'substantia nigra', 'dorsal hippocampus', 'salivary glands', 'saliva', 'heart left ventricle', 'terminal ileum', 'nucleus accumbens', 'normal tissue', 'ewat', 'lung tumor', 'urinary bladder', 'plasma', 'whole kidney', 'cartilage', 'gonad', 'dorsal root ganglion', 'root structure', 'colorectal cancer', 'forebrain', 'embryonic kidney', 'epididymal white adipose tissue', 'abdomen', 'adrenal', 'subventricular zone', 'pituitary gland', 'gingiva', 'medial prefrontal cortex', 'cortex of kidney', 'carpel', 'cardiac ventricle', 'mammary gland/ breast', 'liver tumor tissue', 'whole larvae', 'liver tissue', 'bronchus', 'apical shoot meristem', 'macrophage', 'brain cortex', 'transverse colon', 'primary root', 'breast tumor', 'whole plant', 'plant ovary', 'peritoneum', 'brown adipose tissue (bat)', 'animal ovary', 'brain anterior cingulate cortex', 'pancreatic ductal adenocarcinoma (pdac)', 'leaf sheath', 'peripheral whole blood', '... 11 other values', 'cotyledon', 'visceral fat', 'lung carcinoma', 'sciatic nerve', 'gall bladder', 'normal', 'glomeruli from kidney biopsy', 'breast cancer ductal carcinoma', 'silique', 'umbilical cord', 'blood plasma', 'embryonic cortex', 'brain white matter', 'pistil', 'myxoid liposarcoma'] ,
    ),
    experimental_designs_10 = dict(
        description = "Term for the experimental design",
        _25= ['transcription profiling by array', 'co-expression design', 'compound treatment design', 'disease state design', 'time series design', 'replicate design', 'in vitro design', 'in vivo design', 'genetic modification design', 'unknown experiment design type', 'cell type comparison design', 'stimulus or stress design', 'development or differentiation design', 'reference design', 'strain or line design', 'growth condition design', 'dye swap design', 'organism part comparison design', 'binding site identification design', 'individual genetic characteristics design', 'genotype design', 'comparative genomic hybridization by array', 'ex vivo design', 'all pairs', 'case control design'] ,
        v3=
        ['transcription profiling by array', 'compound treatment design', 'disease state design', 'genetic modification design', 'cell type comparison design', 'stimulus or stress design', 'comparative genomic hybridization by array', 'chip-chip by tiling array', 'case control design', 'genotype design', 'development or differentiation design', 'growth condition design', 'strain or line design', 'time series design', 'binding site identification design', 'transcription profiling by sage', 'organism part comparison design', 'other', 'genotyping design', 'dye swap design', 'methylation profiling by array', 'cellular modification design', 'dose response design', 'transcription profiling by tiling array', 'optimization design'],

        _50= ['chip-chip by array', 'comparative genome hybridization design', 'genotyping design', 'cellular modification design', 'dose response design', 'chip-chip by tiling array', 'transcript identification design', 'species design', 'is expressed design', 'pathogenicity design', 'clinical history design', 'microrna profiling by array', 'sex design', 'chip-seq', 'loop design', 'innate behavior design', 'array platform variation design', 'optimization design', 'transcription profiling by tiling array', 'physiological process design', 'cellular process design', 'genotyping by array', 'high throughput sequencing design', 'other', 'stimulated design type'] ,
        _100= ['quality control testing design', 'observational design', 'methylation profiling by array', 'injury design', 'rnai profiling by array', 'validation by reverse transcription pcr design', 'normalization testing design', 'cell cycle design', 'population based design', 'transcription profiling by sage', 'translational bias design', 'tiling path design', 'rna stability design', 'validation by real time pcr design', 'self vs self design', 'microrna profiling by high throughput sequencing', 'translational design', 'array platform comparison design', 'proteomic profiling by array', 'cell component comparison design', 'organism status design', 'operator variation design', 'expression profiling', 'family based design', 'clip-seq', 'individual comparison design', 'non-targeted transgenic variation design', 'disease state design, ex vivo design, loop design, co-expression design', 'proteomic profiling', 'family history design', 'imprinting design', 'expression design', 'array specific design', 'data integration design', 'growth chamber study', 'rna-seq of coding rna', 'translation profiling', 'age design', 'rna-seq of non coding rna', 'software variation design', 'disease state design, replicate design, co-expression design', 'environmental history design', 'chip-chip', 'antigen profiling', 'expression (type i)', 'cell type comparison design, development or differentiation design, organism part comparison design, ex vivo design, co-expression design', 'transcription profiling by high throughput sequencing', 'rna profiling by high throughput sequencing', 'high throughput sequencing design, binding site identification design', 'time series design, in vitro design, co-expression design'] ,
        _200= ['development or differentiation design, in vitro design, co-expression design', 'cell comparison design, steady state vs stress', 'transcription profiling by array, chip-chip by array', 'prospective cohort study', 'prganism part comparison design, time series design, growth condition design', 'compound treatment design, in vivo design, co-expression design', 'chip-chip by snp array', 'strain abundance design', 'genotype comparison design', 'transcription profiling', 'expression (type ii)', 'faire-seq', 'in vivo', 'transposable element identification design', 'biological replicate', 'time series', 'compound treatment design, in vivo design, reference design, replicate design, co-expression design', 'differential expression', 'cellurar modification', 'genetic modification design, development or differentiation design', 'environmental stress', 'secreted protein identification design', 'stimulus or stress design   in vitro design co-expression design', 'reference design, comparative genome hybridization design, disease state design, co-expression design', 'differential expression design', 'transcription profiling of skeletal muscle after gene transfer by electroporation', 'species comparison design', 'time course design', 'behavioral design type', 'cell type comparison design, in vitro design, co-expression design', 'physiological process design, dye swap design, reference design, replicate design, co-expression design', 'disease state design, in vivo design, co-expression design', 'expression analysis design', 'expression comparison design', 'rip-chip', 'rna sequencing design', 'disease state design, time series design, all pairs, dye swap design, co-expression design', 'in vitro design comparative genome hybridization design', 'disease state design, reference design, comparative genome hybridization design, co-expression design, clinical history design', 'treatment comparison design', 'genetic variation', 'organism', 'strain or line', 'genotyping', 'comaleparative genomalee hybridization design', 'organism part comparison', 'individual genetic characteristics comparison design', 'dna methylation profiling design', 'dye swap', 'genotyping by mass spectrometer', 'compound treatment design human tumour xenografts with the vegfr tyrosine kinase inhibitor cediranib', 'disease state', 'cell type comparision design', 'methylation profiling by high throughput sequencing', 'cellular modification design, reference design, replicate design, co-expression design', 'gene expression', 'genetic modification design,replicate design', 'in vitro design, co-expression design,strain or line design,replicate design,dye swap design', 'cell type comparison design, stimulus or stress design', 'binding site identification design,tiling path design,dye swap design,growth condition design,in vivo design', 'stimulus or stress design, in vivo design, co-expression design', 'compound treatment design, in vitro design, co-expression design', 'compound treatement design', 'agonist comparison design', 'rip', 'compound treatment design, dose response design, in vitro design, replicate design, co-expression design', 'clinical history design, replicate design, co-expression design', 'individual genetic characteristics design, genotyping design', 'behavior design', 'rna-seq', 'compound based treatment', 'operon identification design', 'restriction site identification design', 'differentiation design', 'dye swap design co-expression design', 'compound base treatment', 'development or differentiation design, genetic modification design, ex vivo design, co-expression design', 'growth condition design, in vivo design, co-expression design', 'embryo development design', 'treatment design', 'chip-seq by high throughput sequencing', 'all pairs design', 'disease state design, array platform comparison design, reference design, replicate design, comparative genome hybridization design', 'genetic modification design,individual genetic characteristics design,reference design,replicate design,strain or line design', 'genetic modification design, time series design, replicate design, co-expression design', 'ahcre icd/wt', 'ahcre eyfp/wt', 'cell type design', 'deepsage', 'proteomic profiling by mass spectrometer', 'cgh', 'in vitro design,', 'anoxia treatment design', 'chromatin immunoprecipitation design', 'set for the moss funaria hygrometrica', 'co-expression design, genetic modification design, all pairs', 'rip-chip by array', 'hydroxy-methylation profiling by high throughput sequencing', 'cell line design', 'hardware variation design'] ,
    ),
    assay_by_molecule_14 = dict(
        description = "Assay type specified by molecule",
        _25= ['rna assay', 'dna assay', 'protein assay'] ,
        v3 = ['rna assay', 'dna assay', 'protein assay'],
        _50= [] ,
        _100= [] ,
        _200= [] ,
    ),
    study_type_18 = dict(
        description = "Type of study",
        _25= ['transcription profiling by array', 'rna-seq of coding rna', 'chip-seq', 'rna-seq of non coding rna', 'comparative genomic hybridization by array', 'other', 'methylation profiling by array', 'chip-chip by tiling array', 'methylation profiling by high throughput sequencing', 'unknown experiment type', 'chip-chip by array', 'genotyping by array', 'microrna profiling by array', 'transcription profiling by tiling array', 'rna-seq of coding rna from single cells', 'dna-seq', 'transcription profiling by rt-pcr', 'cell line - high-throughput sequencing', 'animal - high-throughput sequencing', 'proteomic profiling by array', 'transcription profiling by sage', 'high-throughput sequencing', 'baseline', 'atac-seq', 'rnai profiling by array'] ,
        v3= 
        ['transcription profiling by array', 'rna-seq of coding rna', 'chip-seq', 'rna-seq of non coding rna', 'comparative genomic hybridization by array', 'methylation profiling by array', 'other', 'unknown experiment type', 'chip-chip by tiling array', 'methylation profiling by high throughput sequencing', 'chip-chip by array', 'microrna profiling by array', 'transcription profiling by rt-pcr', 'genotyping by array', 'transcription profiling by tiling array', 'dna-seq', 'rna-seq of coding rna from single cells', 'transcription profiling by sage', 'proteomic profiling by array', 'rnai profiling by array', 'clip-seq', 'genotyping by high throughput sequencing', 'microrna profiling by high-throughput sequencing', 'rip-chip by array', 'microrna profiling by high throughput sequencing'],
        _50= ['clip-seq', 'rna-seq of total rna', 'human - high-throughput sequencing', 'cell line - one-color microarray', 'genotyping by high throughput sequencing', 'differential', 'translation profiling', 'animal - one-color microarray', 'human - one-color microarray', 'rip-seq', 'microrna profiling by high throughput sequencing', 'one-color microarray', 'medip-seq', 'plant - high-throughput sequencing', 'animal - single-cell sequencing', 'microrna profiling by high-throughput sequencing', 'human - single-cell sequencing', 'cell line - single-cell sequencing', 'rip-chip by array', 'animal - two-color microarray', 'trajectory', 'mnase-seq', 'antigen profiling', 'scatac-seq', 'tiling path by array'] ,
        _100= ['chip-chip by snp array', 'bisulfite-seq', 'whole chromosome random sequencing', 'transcription profiling by mpss', 'human - two-color microarray', '4c', 'capture-c', 'proteomic profiling by mass spectrometer', 'transcription profiling by high throughput sequencing', 'human - methylation microarray', 'transcription profiling by array, chip-chip by array', 'two-color microarray', 'faire-seq', 'ribo-seq', 'fix: ? by tiling array', 'genotyping by mass spectrometer', 'spatial transcriptomics by high-throughput sequencing', 'hi-c', 'spatial transcriptomics', 'gro-seq'] ,
        _200= [] ,
    ),
)


lengths = [25,50,100,200]
classes = {}
for i, length in enumerate(lengths):

    fields = {}
    for field_name in values:

        description = values[field_name]["description"]
        values_to_include = []
        for j in range(i+1):
            values_to_include = [*values_to_include, *values[field_name][f"_{lengths[j]}"]]

        field_type = Literal[tuple(values_to_include)] # make Literal dynamically by converting to tuple

        field = Field(description = description)
        fields[field_name] = (field_type, field)

    classes[str(length)] = create_model(f'arxpr2_{length}', **fields)

fields = {}
for field_name in values:

    description = values[field_name]["description"]
    values_to_include = values[field_name]["v3"]
    field_type = Literal[tuple(values_to_include)] # make Literal dynamically by converting to tuple

    field = Field(description = description)
    fields[field_name] = (field_type, field)

classes["v3"] = create_model(f'arxpr3', **fields)



def get_shuffled_form_generator(length, only_shuffle_type=False, v3= False):
    """ specify length here, then return pseudo-generator with this already set

    length: number of choices to include, from 25, 50, 100, 200.
    only_shuffle_type: remove all other fields (for ontology retrieval)
    v3: use this to get the arxpr3 values
    """

    def get_shuffled_form(seed=None):
        assert not length is None
        if seed is None:
            return classes["v3" if v3 else str(length)]
        random.seed(seed)
        
        shuffled_fields = {}
        for field_name in values:
            
            if only_shuffle_type and field_name != "study_type_18":
                continue
        
            description = values[field_name]["description"]

            if v3:
                values_to_include = values[field_name]["v3"]
            else:
                values_to_include = []
                j = 0
                while len(values_to_include) < length and len(values[field_name][f"_{lengths[j]}"]): # second term ensures we stop if there are no more values
                    values_to_include = [*values_to_include, *values[field_name][f"_{lengths[j]}"]]
                    j += 1
        
            # shuffle the values
            random.shuffle(values_to_include)
            shuffled_field_type = Literal[tuple(values_to_include)]
        
            field = Field(description = description)
            shuffled_fields[field_name] = (shuffled_field_type, field)
        
        if v3:
            assert length==25
            pydantic_form = create_model(f'arxpr3:{seed}', **shuffled_fields)
        else:
            pydantic_form = create_model(f'arxpr2:{seed}_{length}', **shuffled_fields)
        return pydantic_form
    return get_shuffled_form

Metadata_form = get_shuffled_form_generator(25)()

if __name__ == "__main__":
    import pprint
    pprint.pprint(classes["50"].__fields__["organism_part_5"])
    print("")
    pprint.pprint(get_shuffled_form(4,25).__fields__["organism_part_5"])
    get_shuffled_form(6,25)
    pprint.pprint(get_shuffled_form(4,25).__fields__["organism_part_5"])
