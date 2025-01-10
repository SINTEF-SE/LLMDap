from deeponto.onto import Ontology


def load_ontology(file):
    onto_object = Ontology(file)
    print(type(onto_object))
    print(onto_object)

    return onto_object


def contains_substring(text, substrings=[]):
    """ Checks if the input text contains one of specified strings """

    for s in substrings:
        if s in text:
            return True
    return False


def get_children(onto_object, owl_iri, filter_words, entities=[]):
    """ 
    Given an entitie's OWL IRI (e.g. http://purl.obolibrary.org/obo/OBI_0400044) and set of filter
    keywords, return all annotations of child entities. 

    """
    
    owl_object = onto_object.get_owl_object(owl_iri)
    children = onto_object.get_asserted_children(owl_object, True)
    print('\nchildren:', children)

    # If an end node, return annotation
    if not children:
        annot = onto_object.get_annotations(owl_object)
        filtered_annot = [a for a in annot if not contains_substring(a, filter_words)]
        print('\nfilter_words:', filter_words)
        print('\nannot:', annot)
        print('\nfiltered_annot:', filtered_annot)

        return filtered_annot

    # DFS 
    while children: 
        child_owl = children.pop()
        owl_iri = onto_object.get_iri(child_owl)
        entity = get_children(onto_object, owl_iri, filter_words)
        entities.append(entity)

    return entities


def show_parents(onto_object, owl_iri, keywords = []):
    """ 
    Given an entitie's OWL IRI (e.g. http://purl.obolibrary.org/obo/OBI_0400044) and set of stop
    keywords, match the parent entity matching a specify keyward.

    """

    owl_object = onto_object.get_owl_object(owl_iri)
    curr_annot = onto_object.get_annotations(owl_object)
    print('\ncurr_annot:', curr_annot)

    match = list(set(curr_annot) & set(keywords))
    print('\nmatch:', match)   

    if match:
        children = onto_object.get_asserted_children(owl_object, True)
        print('\nchildren:', len(children), children)

        child_annot = onto_object.get_annotations(children.pop())
        print('\nchild_annot:', child_annot)

        return None    

    parents = onto_object.get_asserted_parents(owl_object, True)
    print('\nparents:', len(parents), parents)

    if parents:
        parent = set([parents.pop()])
        print('\nparent:', parent)
    else:
        return None

    while parent:
        parent_owl = parent.pop()
        parent_annot = onto_object.get_annotations(parent_owl)
        print('\nparent_annot:', parent_annot)

        owl_iri = onto_object.get_iri(parent_owl)
        print('\nowl_iri:', owl_iri)

        return show_parents(onto_object, owl_iri, keywords)        



if __name__ == "__main__":

    # keywords = ['measurement device']
    # filter_words = ['http','PERSON']
    # file = 'obi.owl'

    # measurement_device = 'http://purl.obolibrary.org/obo/OBI_0000832'
    # flow_cytometer = 'http://purl.obolibrary.org/obo/OBI_0400044'
    # low_cytometer_analyzer = 'http://purl.obolibrary.org/obo/OBI_0400008'
    # A10_analyzer = 'http://purl.obolibrary.org/obo/OBI_0400005'



    file = 'efo.owl'
    filter_words = ['http','PERSON']

    # measurement_device = 'http://purl.obolibrary.org/obo/OBI_0000832'
    # flow_cytometer = 'http://purl.obolibrary.org/obo/OBI_0400044'
    # low_cytometer_analyzer = 'http://purl.obolibrary.org/obo/OBI_0400008'
    # A10_analyzer = 'http://purl.obolibrary.org/obo/OBI_0400005'


    onto = load_ontology(file)
    genotype = 'http://www.ebi.ac.uk/efo/EFO_0000513'
    study_design = 'http://www.ebi.ac.uk/efo/EFO_0001426'

    # show_parents(onto, flow_cytometer, keywords)

    annotations = get_children(onto, study_design, filter_words)
    print('\nannotations', len(annotations))
    for a in annotations:
        print(a)



