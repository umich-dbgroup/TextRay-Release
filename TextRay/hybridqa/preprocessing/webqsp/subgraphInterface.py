import networkx as nx

##TODO: use a better representation for pattern matching (query evaluation)

class SubGraph(object):
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    @staticmethod
    def read(triples):
        subgraph = SubGraph()
        for triple in triples:
            src = triple[0]["kb_id"].replace("<","").replace(">","").replace("fb:","")
            rel = triple[1]["rel_id"].replace("<","").replace(">","").replace("fb:","")
            tgt = triple[2]["kb_id"].replace("<","").replace(">","").replace("fb:","")
            subgraph.graph.add_node(src)
            subgraph.graph.add_node(tgt)

            subgraph.graph.add_edge(src, tgt, key=rel)
        return subgraph

    def has_one_step(self, src, p1):
        for _, _, rel in self.graph.edges([src], keys=True):
            if rel == p1:
                return True
        return False

    def has_two_step(self, src, p1, p2):
        for _, inter, rel1 in self.graph.edges([src], keys=True):
            if rel1 == p1:
                for _, _, rel2 in self.graph.edges([inter], keys=True):
                    if rel2 == p2:
                        return True
        return False