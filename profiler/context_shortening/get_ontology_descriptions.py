from owlready2 import get_ontology


def get_subontology(
            mode, # label or description
            ontology_path = "../ontologies/efo.owl",  # Path to your OWL file
            top_level_node_iri = "http://purl.obolibrary.org/obo/OBI_0000070", # node labeled "assay"
            node_string_filter = "",
            
            ):
    """ get the description and/or labels of every node that is a descendant of the "assay" node (or another specified, on the efo ontology"""

    assert mode in ["label","description", "both"]

    # Load the ontology using OWLReady2 and extract concept labels
    ontology = get_ontology(ontology_path).load()

    # Prune ontology
    top_level_node = ontology.search_one(iri = top_level_node_iri)
    descendants = top_level_node.descendants()

    # BTO ontology file does not include all relations (quite few actually), but since we want the top node we workaround:
    if ontology_path.endswith("bto.owl"):
        assert top_level_node_iri.endswith("BTO_0000000")
        descendants = ontology.classes()

    # Extract labels for each concept in the ontology
    ontology_descriptions = []
    for node in descendants:
        if not node_string_filter in str(node):
            continue
        node_descriptions = node.IAO_0000115

        if len(node_descriptions) and mode in ["description", "both"]:
            ontology_descriptions.append(node_descriptions[-1]) # in 4 cases for the assay subontology, there are 2 or 3 descriptions. In all cases the last one is best (they are very similar)
        if not (len(node_descriptions) and mode == "description"):
            if len(node.label):
                ontology_descriptions.append(node.label[0]) # there are 2 labels on some cases, (the if editors prefered label is different i think), but not among these ones without description
            else:
                print("INFO: ontology node missing label:", node, node.IAO_0000115)
    #print(f"Ontology descriptions tail: {ontology_descriptions[-4:]}")
    return ontology_descriptions

SUBTREE_BY_FIELDNAME = {
        "hardware_4" :(
            "../ontologies/obi.owl",
            "http://purl.obolibrary.org/obo/OBI_0400103"
            ),
        "organism_part_5" :(
            "../ontologies/bto.owl",
            "http://purl.obolibrary.org/obo/BTO_0000000"
            ),
        "experimental_designs_10" :(
            "../ontologies/efo.owl", 
            "http://www.ebi.ac.uk/efo/EFO_0001426"
            ),
        "assay_by_molecule_14" :(
            "../ontologies/efo.owl", 
            "http://www.ebi.ac.uk/efo/EFO_0002772"
            ),
        "study_type_18" :(
            "../ontologies/efo.owl",
            "http://purl.obolibrary.org/obo/OBI_0000070", # node labeled "assay"
            "EFO", # for some reason terms from obi that are not in efo are included by default. We only want the EFO ones.
            )
        }

def get_subontology_for_field(
        mode,
        fieldname,
        ):

    return get_subontology(mode, *SUBTREE_BY_FIELDNAME[fieldname])


if __name__ == "__main__":

    for fieldname in SUBTREE_BY_FIELDNAME:
        for mode in ["label", "description", "both"]:
            desc = get_subontology_for_field(mode, fieldname)
            print(fieldname)
            print(len(desc))
            #print(len(set(desc)))
            #print(desc[:10])
            #print(desc[-10:])
            print("")


