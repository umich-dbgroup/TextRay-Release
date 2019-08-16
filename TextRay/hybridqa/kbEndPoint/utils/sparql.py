from SPARQLWrapper import SPARQLWrapper, JSON, N3
import logging
from dataModels.kbModels import Relation, Entity, Value, Node
import time
from string import Template
import pandas as pd


class sparqlUtils:
    def __init__(self, host="141.212.110.80", port="3093"):
        logging.basicConfig(filename='sparql.log', level=logging.INFO)
        url = "http://" + host + ":" + port + "/sparql"
        self.sparqlExecutor1 = SPARQLWrapper(url)
        self.sparqlExecutor1.setReturnFormat(JSON)
        self.sparqlExecutor1.setTimeout(10)
        self.ENCODING = 'utf-8'

        self.one_hop = Template('''
                                PREFIX ns: <http://rdf.freebase.com/ns/>
                                SELECT DISTINCT ?p1 (count(?x) AS ?ct)
                                WHERE {
                                    ${e1} ?p1 ${e2} .
                                    ${f}   
                                }
                                ''')

        self.eval_one_hop = Template('''
                                                PREFIX ns: <http://rdf.freebase.com/ns/>
                                                SELECT DISTINCT ?x
                                                WHERE {
                                                    ${e1} ns:${p1} ${e2} .
                                                    ${f}   
                                                }
                                                ''')

        self.two_hop = Template('''
                                    PREFIX ns: <http://rdf.freebase.com/ns/>
                                    SELECT DISTINCT ?p1 ?p2 (count(?x) AS ?ct)
                                    WHERE {
                                        ${e1} ?p1 ?y .
                                        ?y ?p2 ${e2} .
                                        FILTER NOT EXISTS {?y ns:type.object.name ?yname} .
                                        ${f}
                                    }
                                    ''')

        self.eval_two_hop = Template('''
                                            PREFIX ns: <http://rdf.freebase.com/ns/>
                                            SELECT DISTINCT ?x
                                            WHERE {
                                                ${e1} ns:${p1} ?y .
                                                ?y ns:${p2} ${e2} .
                                                FILTER NOT EXISTS {?y ns:type.object.name ?yname} .
                                                ${f}
                                            }
                                            ''')

        self.eval_cvt = Template('''
                                                    PREFIX ns: <http://rdf.freebase.com/ns/>
                                                    SELECT DISTINCT ?y
                                                    WHERE {
                                                        ns:${e1} ns:${p1} ?y .
                                                        ?y ns:${p2} ns:${e2} .
                                                        FILTER NOT EXISTS {?y ns:type.object.name ?yname} .
                                                        ${f}
                                                    }
                                                    ''')

        self.counts = Template('''
                                    PREFIX ns: <http://rdf.freebase.com/ns/>
                                    SELECT DISTINCT (count(?x) AS ?ct)
                                    WHERE {
                                        ${e1} ${var} ${e2}.
                                        ?x ns:type.object.name ?name .
                                        ${f}   
                                    }
                                    ''')

        self.one_hop_connecting_path = Template('''
                                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                                        SELECT DISTINCT ?p1 
                                                        WHERE {
                                                            ${r} .
                                                            ?ie ns:type.object.name ?name .
                                                            ?ie ?p1 ns:${ans} .
                                                            FILTER ( ns:${e} != ?ie ) .
                                                            FILTER ( ns:${ans} != ?ie ) .
                                                            ${f} 
                                                        } LIMIT 500
                                                ''')

        self.two_hop_connecting_path = Template('''
                                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                                        SELECT DISTINCT ?p1 ?p2
                                                        WHERE {
                                                            ${r} .
                                                            ?ie ns:type.object.name ?name .
                                                            ?ie ?p1 ?y1 .
                                                            ?y1 ?p2 ns:${ans}
                                                            FILTER NOT EXISTS {?y1 ns:type.object.name ?y1name} .
                                                            FILTER ( ns:${e} != ?ie ) .
                                                            FILTER ( ns:${ans} != ?ie ) .
                                                            ${f}   
                                                        } LIMIT 500
                                                ''')

        self.eval_connecting_path = Template('''
                                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                                        SELECT DISTINCT ?x
                                                        WHERE {
                                                            ${r}
                                                            ?ie ${var2} ?x .
                                                            ?x ns:type.object.name ?name .
                                                            FILTER ( ns:${e} != ?x ) .
                                                            FILTER ( ns:${e} != ?ie ) .
                                                            FILTER ( ?ie != ?x ) .
                                                            ${f}   
                                                        } LIMIT 500
                                                ''')

        self.constraints_cvt = Template('''
                                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                                        SELECT DISTINCT ?rc
                                                        WHERE {
                                                            ${e1} ns:${p1} ?y .
                                                            ?y ns:${p2} ${e2} .
                                                            ?x ns:type.object.name ?xname .
                                                            ?y ?rc ns:${c} .
                                                            FILTER NOT EXISTS {?y ns:type.object.name ?yname} .
                                                            FILTER ( ns:${c} != ?x ) .
                                                            ${f}
                                                        }
                                                                ''')

        self.eval_constraints = Template('''
                                                                PREFIX ns: <http://rdf.freebase.com/ns/>
                                                                SELECT DISTINCT ?x
                                                                WHERE {
                                                                    ${e1} ${var} ${e2} .
                                                                    ?x ns:type.object.name ?xname .
                                                                    ${f}
                                                                }
                                                                        ''')

        self.eval_constraints_named = Template('''
                                                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                                                        SELECT DISTINCT ?xname
                                                                        WHERE {
                                                                            ${e1} ${var} ${e2} .
                                                                            ?x ns:type.object.name ?xname .
                                                                            ${f}
                                                                        }
                                                                                ''')

        self.named_ans_constraints = Template('''
                                                                PREFIX ns: <http://rdf.freebase.com/ns/>
                                                                SELECT DISTINCT ?c
                                                                WHERE {
                                                                    ${e1} ${p} ${e2} .
                                                                    ?x ns:type.object.name ?xname .
                                                                    ?x ns:${rc} ?c .
                                                                    ?c ns:type.object.name ?cname .
                                                                    FILTER ( ${e1} != ?c ) .
                                                                    FILTER ( ${e2} != ?c ) .
                                                                    ${f}
                                                                }
                                                                        ''')

        self.constraint_rels = Template('''
                                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                                        SELECT DISTINCT ?rc
                                                        WHERE {
                                                            ${e1} ${p} ${e2} .
                                                            ?x ?rc ns:${c} . 
                                                            ?x ns:type.object.name ?name .
                                                            FILTER ( ?x != ns:${c} ) .
                                                            ${f}  
                                                        }
                                                        ''')

        self.ans_type_constraints_webq = Template('''
                                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                                        SELECT DISTINCT ?c
                                                        WHERE {
                                                            ?e ${var}
                                                            ?x ns:${rc} ?c . 
                                                            ?x ns:type.object.name ?name .
                                                            ?e ns:type.object.name ?ename .
                                                            FILTER ( ?e != ?x ) .
                                                            ${f}   
                                                        }
                                                        ''')

        self.cvt_constraints_webq = Template('''
                                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                                        SELECT DISTINCT ?rc ?c
                                                        WHERE {
                                                            ?e ns:${p1} ?y .
                                                            ?y ns:${p2} ?x .
                                                            ?x ns:type.object.name ?xname .
                                                            ?e ns:type.object.name ?ename .
                                                            ?y ?rc ?c .
                                                            ?c ns:type.object.name ?cname .
                                                            FILTER NOT EXISTS {?y ns:type.object.name ?yname} .
                                                            FILTER ( ?e != ?x ) .
                                                            ${f}
                                                        }
                                                                ''')

        self.entity_type = Template('''
                                                PREFIX ns: <http://rdf.freebase.com/ns/>
                                                SELECT DISTINCT ?xname
                                                WHERE {
                                                    ns:${e} ns:common.topic.notable_types ?x .
                                                    ?x ns:type.object.name ?xname .
                                                    FILTER(langMatches(lang(?xname), 'en'))
                                                } LIMIT 10
                                                ''')

        self.one_hop_conjunction = Template('''
                                                                PREFIX ns: <http://rdf.freebase.com/ns/>
                                                                SELECT DISTINCT ?p1 
                                                                WHERE {
                                                                    ${r} .
                                                                    ?x ns:type.object.name ?name .
                                                                    ?x ?p1 ${val} .
                                                                    FILTER ( ns:${e} != ?x ) .
                                                                    ${f} 
                                                                } LIMIT 500
                                                        ''')

        self.two_hop_conjunction = Template('''
                                                                PREFIX ns: <http://rdf.freebase.com/ns/>
                                                                SELECT DISTINCT ?p1 ?p2
                                                                WHERE {
                                                                    ${r} .
                                                                    ?x ns:type.object.name ?name .
                                                                    ?x ?p1 ?y1 .
                                                                    ?y1 ?p2 ${val}
                                                                    FILTER NOT EXISTS {?y1 ns:type.object.name ?y1name} .
                                                                    FILTER ( ns:${e} != ?x ) .
                                                                    ${f}   
                                                                } LIMIT 500
                                                        ''')

        self.eval_conjunction = Template('''
                                                                PREFIX ns: <http://rdf.freebase.com/ns/>
                                                                SELECT DISTINCT ?x
                                                                WHERE {
                                                                    ${r}
                                                                    ?x ${var2} ${val} . 
                                                                    ?x ns:type.object.name ?name .
                                                                    FILTER ( ns:${e} != ?x ) .
                                                                    ${f}   
                                                                } LIMIT 500
                                                        ''')



    def execute(self, query, executorID=1):
        '''
        :param qeury:
        :return: parsed json format
        '''
        results = {'results': {'bindings': None}}
        if (executorID == 1):
            self.sparqlExecutor1.setQuery(query)
            #results = self.sparqlExecutor1.query().convert()
            try:
                self.sparqlExecutor1.setQuery(query)
                results = self.sparqlExecutor1.query().convert()
            except:
              return results
        return results

    def evaluate_sparql(self, query):
        '''
        :param query: original query (replaced variable with *)
        :return: parsed json format of the response
        '''
        idx1 = query.index("SELECT")
        idx2 = query.index("WHERE")
        subgraph_query = query.replace(query[idx1 + 6: idx2], ' * ')
        results = self.execute(subgraph_query)
        return results

    def get_names(self, mid):
        '''
        :param mid:
        :return: names
        '''
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?o WHERE { ns:" + mid + " ns:type.object.name ?o . FILTER(langMatches(lang(?o), 'en')) }"
        names = self.execute(query)[u'results'][u'bindings']
        if names is not None and len(names) > 0:
            return names[0][u'o'][u'value']
        return None


    def get_names_alias(self, mid):
        '''
        :param mid:
        :return: list of names or alias (string)
        '''
        prefixed = "ns:" + mid
        name_surface = []
        name_relations = ["ns:type.object.name","ns:common.topic.alias","ns:base.schemastaging.context_name.abbreviation", "ns:base.schemastaging.context_name.abbreviation"]
        for name_relation in name_relations:
            query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?o WHERE {" + prefixed + " " + name_relation + " ?o . FILTER(langMatches(lang(?o), 'en')) }"
            query_results = self.execute(query)[u'results'][u'bindings']
            for query_result in query_results:
                name_surface.append(query_result[u'o'][u'value'])
        return name_surface


    def getRelationNamespace(self, relation):
        results = []
        cleaned_relation = self.remove_uri_prefix(relation)
        all_tokens = cleaned_relation.split('.')
        tokens = all_tokens[:len(all_tokens) - 1]
        for token in tokens:
            results = results + token.split('_')
        return results

    def getRelationTokens(self, relation, only_namespace=False, domain=True):
        '''
        :param relation: standard KB relation, starts with http
        :param only_namespace: only consider namespace of the relation
        :return: list of tokens of relations, without domain
        '''
        results = []
        cleaned_relation = self.remove_uri_prefix(relation)
        if only_namespace:
            all_tokens = cleaned_relation.split('.')
            tokens = all_tokens[:len(all_tokens) - 1]
            for token in tokens:
                results = results + token.split('_')
        elif not domain:
            rel_name = cleaned_relation.split('.')[-1]
            results =  rel_name.split("_")
            return results
        else:
            tokens = cleaned_relation.split('.')
            for token in tokens:
                results = results + token.split('_')
        return results

    def get_entity_mids(self, query):
        '''
        :param query: sparql annotation
        :return: list of mids, for example: m.xxx
        '''
        entities = set()
        sparql_tokens = query.replace('\n', ' ').replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').strip().split()
        for token in sparql_tokens:
            if token.strip().startswith("ns:m.") or token.strip().startswith("ns:g."):
                if token[-1] != ')':
                    entities.add(token[3:].encode(self.ENCODING))
        return list(entities)

    def get_entity_types(self, mid):
        query = self.entity_type.substitute(e=mid)
        results = self.execute(query)['results']['bindings']
        if results is None:
            return []
        return [r["xname"]["value"] for r in results]

    def exists(self, mid):
        '''
        :param mid:
        :return: if it exists in knowledge base
        '''
        res = self.getoneHop_Relation(mid, filter_literal=False, filter_type=False)
        return len(res) != 0

    def one_hop_expansion(self, mid, relations_to_filter=None):
        expansion_cands = self.one_hop_expansion0(mid, "?x", relations_to_filter=relations_to_filter)
        as_tuples = [(r["p1"], r["ct"]) for r in expansion_cands]
        return set(as_tuples)

    def one_hop_rev_expansion(self, mid, relations_to_filter=None, max_ct=2000):
        expansion_cands = self.one_hop_expansion0("?x", mid, relations_to_filter=relations_to_filter)
        as_tuples = [(r["p1"], r["ct"]) for r in expansion_cands if float(r["ct"]) < max_ct]
        return set(as_tuples)

    def one_hop_expansion0(self, src, tgt, relations_to_filter=None):
        filter = self.__get_pred_filter__("?p1", relations_to_filter=relations_to_filter) + self.__get_entity_filter__("?x")
        e1 = src
        if not e1.startswith("?"): e1 = "ns:" + e1
        e2 = tgt
        if not e2.startswith("?"): e2= "ns:" + e2
        query = self.one_hop.substitute(e1=e1, e2=e2, f=filter)
        results = self.execute(query)['results']['bindings']
        return self.parse_rel_result(results, relations_to_filter=relations_to_filter)


    def eval_one_hop_expansion(self, mid, rel1):
        return self.eval_one_hop_expansion0(mid, "?x", rel1)

    def eval_one_hop_rev_expansion(self, mid, rel1):
        return self.eval_one_hop_expansion0("?x", mid, rel1)

    def eval_one_hop_expansion0(self, src, tgt, rel1):
        e1 = src
        if not e1.startswith("?"): e1 = "ns:" + e1
        e2 = tgt
        if not e2.startswith("?"): e2 = "ns:" + e2
        filter = self.__get_entity_filter__("?x")
        filter += "FILTER(" + e1 + "!=" + e2 + " ) ."
        query = self.eval_one_hop.substitute(e1=e1, e2=e2, f=filter, p1=rel1)
        # print (query)
        results = self.execute(query)['results']['bindings']
        ans = []
        if results is not None and len(results) > 0:
            ans = [self.remove_uri_prefix(r["x"]["value"]) for r in results]
        return ans

    def two_hop_expansion(self, mid, relations_to_filter=None):
        expansion_cands = self.two_hop_expansion0(mid, "?x", relations_to_filter=relations_to_filter)
        as_tuples = [(r["p1"], r["p2"], r["ct"]) for r in expansion_cands]
        return set(as_tuples)

    def two_hop_rev_expansion(self, mid, relations_to_filter=None, max_ct=2000):
        expansion_cands = self.two_hop_expansion0("?x", mid, relations_to_filter=relations_to_filter)
        as_tuples = [(r["p1"], r["p2"], r["ct"]) for r in expansion_cands if float(r["ct"]) < max_ct]
        return set(as_tuples)

    def two_hop_expansion0(self, src, tgt, relations_to_filter=None):
        e1 = src
        if not e1.startswith("?"): e1 = "ns:" + e1
        e2 = tgt
        if not e2.startswith("?"): e2 = "ns:" + e2
        filter = self.__get_pred_filter__("?p1", relations_to_filter=relations_to_filter) \
                 + self.__get_pred_filter__("?p2", relations_to_filter=relations_to_filter) \
                 + self.__get_entity_filter__("?x") \
                 + "FILTER(" + e1 + "!=" + e2 + " ) ."
        query = self.two_hop.substitute(e1=e1, e2=e2, f=filter)
        results = self.execute(query)['results']['bindings']
        return self.parse_rel_result(results, relations_to_filter=relations_to_filter)

    def eval_two_hop_expansion(self, mid, rel1, rel2):
        return self.eval_two_hop_expansion0(mid, "?x", rel1, rel2)

    def eval_two_hop_rev_expansion(self, mid, rel1, rel2):
        return self.eval_two_hop_expansion0("?x", mid, rel1, rel2)

    def eval_two_hop_expansion0(self, src, tgt, rel1, rel2):
        e1 = src
        if not e1.startswith("?"): e1 = "ns:" + e1
        e2 = tgt
        if not e2.startswith("?"): e2 = "ns:" + e2
        filter = self.__get_entity_filter__("?x")
        query = self.eval_two_hop.substitute(e1=e1, e2=e2, p1=rel1, p2=rel2, f=filter)
        results = self.execute(query)['results']['bindings']
        ans = []
        if results is not None and len(results) > 0:
            ans = [self.remove_uri_prefix(r["x"]["value"]) for r in results]
        return ans

    def get_cvt_nodes(self, src, tgt, rel1, rel2):
        query = self.eval_cvt.substitute(e1=src, e2=tgt, p1=rel1, p2=rel2, f='')
        results = self.execute(query)['results']['bindings']
        ans = []
        if results is not None and len(results) > 0:
            ans = [self.remove_uri_prefix(r["y"]["value"]) for r in results]
        return ans

    def get_node_constraints(self, src, relation):
        e1 = "ns:" + src
        e2 = "?x"
        query = self.eval_one_hop.substitute(e1=e1, e2=e2, f='', p1=relation)
        ans = []
        results = self.execute(query)['results']['bindings']
        if results is not None and len(results) > 0:
            ans = [self.remove_uri_prefix(r["x"]["value"]) for r in results]
        return ans


    def get_interim_size(self, topic_entity, relation_path, is_reverse):
        var = self.relation_path_to_query(relation_path)
        filter = ""
        if len(relation_path) == 2: filter = "FILTER NOT EXISTS {?y ns:type.object.name ?yname} ."
        if is_reverse:
            query = self.counts.substitute(var=var, e1="?x", e2="ns:"+topic_entity, f=filter)
        else:
            query = self.counts.substitute(var=var, e1="ns:" + topic_entity, e2="?x", f=filter)
        results = self.execute(query)['results']['bindings']
        if results is None or len(results) == 0:
            return 0
        return results[0]["ct"]["value"]


    def get_conjunction_path(self, topic_entity, relation_path, named_entity_value):
        relations = relation_path['relations']
        var = self.relation_path_to_query(relations)
        if relation_path["is_reverse"]:
            var = "?x " + var + " ns:" + topic_entity + " "
        else:
            var = "ns:" + topic_entity + " " + var + " ?x "
        filter = self.__get_pred_filter__("?p1")
        named_entity_value = "\"" + named_entity_value
        if named_entity_value.endswith("^^xsd:dateTime"):
            named_entity_value = named_entity_value.replace("^^", "\"^^")
        else:
            named_entity_value += "\""
        query = self.one_hop_conjunction.substitute(r=var, val=named_entity_value, f=filter, e=topic_entity)
        #print(query)
        paths = []
        results = self.execute(query)['results']['bindings']
        rels = self.parse_rel_result(results)
        paths += [{"relations": [r["p1"]], "is_reverse": False} for r in rels]
        if len(paths) > 0:
            return paths
        return paths
        '''disabling because this is terribly slow!'''
        # filter = self.__get_pred_filter__("?p1") + self.__get_pred_filter__("?p2")
        # query = self.two_hop_conjunction.substitute(e=topic_entity, r=var, val=named_entity_value, f=filter)
        # print(query)
        # results = self.execute(query)['results']['bindings']  # for time outs
        # rels = self.parse_rel_result(results)
        # paths += [{"relations": [r["p1"], r["p2"]], "is_reverse": False} for r in rels]
        # return paths

    def evaluate_conjunction_path(self, topic_entity, relation_path1, relation_path2, named_entity_value):
        filter = self.__get_entity_filter__("?x")
        relations = relation_path1['relations']
        var1 = self.relation_path_to_query(relations)
        if relation_path1["is_reverse"]: var1 = "?x " + var1 + " ns:" + topic_entity + " ."
        else: var1 = "ns:" + topic_entity + " " + var1 + " ?x ."

        relations2 = relation_path2["relations"]
        var2 = self.relation2_path_to_query(relations2)
        named_entity_value = "\"" + named_entity_value
        if named_entity_value.endswith("^^xsd:dateTime"):
            named_entity_value = named_entity_value.replace("^^", "\"^^")
        else:
            named_entity_value += "\""
        query = self.eval_conjunction.substitute(r=var1,var2=var2,f=filter,e=topic_entity, val=named_entity_value)
        # print(query)
        results = self.execute(query)['results']['bindings']
        ans = []
        if results is not None and len(results) > 0:
            for r in results:
                ans.append(self.remove_uri_prefix(r["x"]["value"]))
        return ans


    # do not consider reverse directions from interim entity to ans entity
    # only considering forward relations from interim entity to ans entity
    def get_connecting_path(self, topic_entity, relation_path, answer_entity):
        if topic_entity == answer_entity:
            return []
        relations = relation_path['relations']
        var = self.relation_path_to_query(relations)
        if relation_path["is_reverse"]:
            var = "?ie " + var + " ns:" + topic_entity + " "
        else:
            var = "ns:" + topic_entity + " " + var + " ?ie "
        filter = self.__get_pred_filter__("?p1")
        query = self.one_hop_connecting_path.substitute(r=var, ans=answer_entity, f=filter, e=topic_entity)
        # print query
        paths = []
        results = self.execute(query)['results']['bindings']
        rels = self.parse_rel_result(results)
        paths += [{"relations": [r["p1"]], "is_reverse": False} for r in rels]
        if len(paths) > 0:
            return paths

        filter = self.__get_pred_filter__("?p1") + self.__get_pred_filter__("?p2")
        query = self.two_hop_connecting_path.substitute(e=topic_entity, r=var, ans=answer_entity, f=filter)
        # print(query)
        results = self.execute(query)['results']['bindings']  # for time outs
        rels = self.parse_rel_result(results)
        paths += [{"relations": [r["p1"], r["p2"]], "is_reverse": False} for r in rels]
        return paths

    def evaluate_connecting_path(self, topic_entity, relation_path1, relation_path2):
        filter = self.__get_entity_filter__("?x")
        relations = relation_path1['relations']
        var1 = self.relation_path_to_query(relations)
        if relation_path1["is_reverse"]: var1 = "?ie " + var1 + " ns:" + topic_entity + " ."
        else: var1 = "ns:" + topic_entity + " " + var1 + " ?ie ."

        relations2 = relation_path2["relations"]
        var2 = self.relation2_path_to_query(relations2)

        query = self.eval_connecting_path.substitute(r=var1,var2=var2,f=filter,e=topic_entity)
        # print(query)
        results = self.execute(query)['results']['bindings']
        ans = []
        if results is not None and len(results) > 0:
            for r in results:
                ans.append(self.remove_uri_prefix(r["x"]["value"]))
        return ans

    # this takes into account the directiont too!
    def get_all_cvt_constraints(self, src, relations, is_reverse, constraint_mid):
        constraint_rels = set()
        if len(relations) != 2: return constraint_rels
        e1 = "ns:" + src
        e2 = "?x"
        filter = "FILTER (" + e1 + "!= ?x ) ."
        if is_reverse:
            e1 = "?x"
            e2 = "ns:" + src
            filter = "FILTER (" + e2 + "!= ?x ) ."
        query = self.constraints_cvt.substitute(e1=e1, e2=e2, p1=relations[0], p2=relations[1], f=filter, c=constraint_mid)
        # print(query)
        results = self.execute(query)['results']['bindings']
        if results is None or len(results) == 0:
            return constraint_rels
        for r in results:
            p1 = self.remove_uri_prefix(r["rc"]["value"])
            constraint_rels.add(p1)
        return constraint_rels

    def eval_all_constraints_named(self, src, relation_path, constraints, is_reverse):
        filter = self.__get_entity_filter__("?x")
        filter += "FILTER ( ?x != ns:" + src + " ) .\n"
        cvt_filter = ''' FILTER NOT EXISTS {?y ns:type.object.name ?yname}.\n'''
        has_cvt = False
        for constraint in constraints:
            # print("{} has {} constraint".format(relation_path, constraint))
            c_mid = constraint.mid
            c_rel = constraint.relation
            filter += "FILTER ( ?x != ns:" + c_mid + " ) .\n"
            if constraint.is_ans_constraint:
                filter += "?x " + "ns:" + c_rel + " " + "ns:" + c_mid + " . \n"
            else :
                filter += "?y " + "ns:" + c_rel + " "  + "ns:" + c_mid + " . \n"
                if not has_cvt:
                    has_cvt = True
                    filter += cvt_filter

        rel_var = self.relation_path_to_query(relation_path)
        e1 = "ns:" + src
        e2 = "?x"
        if is_reverse:
            e1 = "?x"
            e2 = "ns:" + src
        query = self.eval_constraints_named.substitute(e1=e1, e2=e2, var=rel_var, f=filter)
        #print(query)
        ans = []
        results = self.execute(query)['results']['bindings']
        if results is not None and len(results) > 0:
            for r in results:
                ans.append(self.remove_uri_prefix(r["xname"]["value"]))
        return ans

    def eval_all_constraints(self, src, relation_path, constraints, is_reverse):
        filter = self.__get_entity_filter__("?x")
        filter += "FILTER ( ?x != ns:" + src + " ) .\n"
        cvt_filter = ''' FILTER NOT EXISTS {?y ns:type.object.name ?yname}.\n'''
        has_cvt = False
        for constraint in constraints:
            # print("{} has {} constraint".format(relation_path, constraint))
            c_mid = constraint.mid
            c_rel = constraint.relation
            filter += "FILTER ( ?x != ns:" + c_mid + " ) .\n"
            if constraint.is_ans_constraint:
                filter += "?x " + "ns:" + c_rel + " " + "ns:" + c_mid + " . \n"
            else :
                filter += "?y " + "ns:" + c_rel + " "  + "ns:" + c_mid + " . \n"
                if not has_cvt:
                    has_cvt = True
                    filter += cvt_filter

        rel_var = self.relation_path_to_query(relation_path)
        e1 = "ns:" + src
        e2 = "?x"
        if is_reverse:
            e1 = "?x"
            e2 = "ns:" + src
        query = self.eval_constraints.substitute(e1=e1, e2=e2, var=rel_var, f=filter)
        #print(query)
        ans = []
        results = self.execute(query)['results']['bindings']
        if results is not None and len(results) > 0:
            for r in results:
                ans.append(self.remove_uri_prefix(r["x"]["value"]))
        return ans

    def get_ans_constraint_candidates(self, src, relation_path, constraint_relations_to_filter=None, is_reverse=False):
        if constraint_relations_to_filter is None:
            return []
        path = self.relation_path_to_query(relation_path)
        filter = ""
        if len(relation_path) == 2: filter += ''' FILTER NOT EXISTS {?y ns:type.object.name ?yname}.\n'''
        e1 = "ns:" + src
        e2 = "?x"
        if is_reverse:
            e1 = "?x"
            e2 = "ns:" + src
        ans_types = {}
        for constraint_relation in constraint_relations_to_filter:
            query = self.named_ans_constraints.substitute(e1=e1, e2=e2, p=path, rc=constraint_relation, f=filter)
            # print(query)
            results = self.execute(query)['results']['bindings']
            if results is None: continue
            for r in results:
                c = self.remove_uri_prefix(r["c"]["value"])
                ans_types[c] = constraint_relation
        return ans_types

    def get_ans_constraint_rel(self, src, relation_path, constraint_mid, relations_to_filter=None, constraint_relations_to_filter=None, is_reverse=False):
        constraint_rels = set()
        filter = self.__get_filter__("?rc", constraint_relations_to_filter)
        if len(relation_path) == 2: filter += ''' FILTER NOT EXISTS {?y ns:type.object.name ?yname}.\n'''
        e1 = "ns:" + src
        e2 = "?x"
        if is_reverse:
            e1 = "?x"
            e2 = "ns:" + src
        path = self.relation_path_to_query(relation_path)
        query = self.constraint_rels.substitute(e1=e1,e2=e2,p=path,f=filter,c=constraint_mid)
        # print(query)
        results = self.execute(query)['results']['bindings']
        if results is None: return constraint_rels
        for r in results:
            p1 = self.remove_uri_prefix(r["rc"]["value"])
            constraint_rels.add(p1)
        return constraint_rels

    #for webqsp
    def get_skeleton_constraint_candidates(self, relation_path, constraint_relations_to_filter):
        print relation_path
        constraint_dict = {}
        if relation_path is None or len(relation_path) == 0:
            return constraint_dict
        if relation_path == 2:
            # cvt constraints
            query = self.cvt_constraints_webq.substitute(p1 = relation_path[0], p2 = relation_path[1], f='')
            results = self.execute(query)['results']['bindings']
            if results is not None:
                for r in results:
                    print ('cvt constraint \t'  + str(r))
                    rc = self.remove_uri_prefix(r["rc"]["value"])
                    c = self.remove_uri_prefix(r["c"]["value"])
                    relations = []
                    if c in constraint_dict: relations = constraint_dict[c]
                    relations.append(rc)
                    constraint_dict[c] = relations

        # ans constraints
        if relation_path == 2:
            var = "ns:" + relation_path[0] + " ?y .\n" + "?y ns:" + relation_path[1] + "?x . \n"
            filter = "FILTER NOT EXISTS {?y ns:type.object.name ?yname} ."
        else:
            var = "ns:" + relation_path[0] + " ?x .\n"
            filter = ""
        for constraint_relation in constraint_relations_to_filter:
            query = self.ans_type_constraints_webq.substitute(var=var,rc=constraint_relation,f=filter)
            results = self.execute(query)['results']['bindings']
            if results is None: continue
            for r in results:
                print ('ans constraint \t' + str(r))
                c = self.remove_uri_prefix(r["c"]["value"])
                relations = []
                if c in constraint_dict: relations = constraint_dict[c]
                relations.append(constraint_relation)
                constraint_dict[c] = relations
        print('constraints found \t' + str(len(constraint_dict)))
        return constraint_dict

    # def one_hop_path(self, src, tgt, relations_to_filter=None):
    #     filter = self.__get_pred_filter__("?p1", relations_to_filter=relations_to_filter)
    #     query = self.template5.substitute(e1=src, e2=tgt, f=filter)
    #     results = self.execute(query)['results']['bindings']
    #     paths = []
    #     if results is None:
    #         return set(paths)
    #     for r in results:
    #         p1 = self.remove_uri_prefix(r["p1"]["value"])
    #         if relations_to_filter is not None:
    #             if p1 not in relations_to_filter:
    #                continue
    #             paths.append(p1)
    #         else:
    #             paths.append(p1)
    #     return set(paths)
    #
    #
    # def two_hop_path(self, src, tgt, relations_to_filter=None):
    #     filter = self.__get_pred_filter__("?p1", relations_to_filter=relations_to_filter) \
    #              + self.__get_pred_filter__("?p2", relations_to_filter=relations_to_filter)
    #     query = self.template6.substitute(e1=src, e2=tgt, f=filter)
    #
    #     results = self.execute(query)['results']['bindings']
    #     paths = []
    #     if results is None:
    #         return set(paths)
    #     for r in results:
    #         p1 = self.remove_uri_prefix(r["p1"]["value"])
    #         p2 = self.remove_uri_prefix(r["p2"]["value"])
    #         if relations_to_filter is not None:
    #             if p1 not in relations_to_filter or p2 not in relations_to_filter:
    #                 continue
    #             paths.append((p1, p2))
    #         else:
    #             paths.append((p1, p2))
    #     return set(paths)
    #
    # def one_step(self, mid, relations_to_filter=None):
    #     '''
    #     :param mid: id of the topic entity
    #     :return: response
    #     '''
    #     if mid[0:2] != 'm.':
    #         return set()
    #     prefixed = "ns:" + mid
    #     node_data = {"type" : "uri", "value": "http://rdf.freebase.com/ns/" + mid}
    #     query1 = """PREFIX ns: <http://rdf.freebase.com/ns/>
    #                     SELECT DISTINCT * WHERE {
    #                                          """ + prefixed + """ ?p ?o . """ \
    #              + "FILTER(?o != " + prefixed + ")" \
    #              + self.__get_pred_filter__('?p', relations_to_filter=relations_to_filter) \
    #              + self.__get_entity_filter__("?o") + "}"
    #     results1 = self.execute(query1)['results']['bindings']
    #     triples = set()
    #     if results1 is not None and len(results1) > 0:
    #         for r in results1:
    #             triples.add(Relation.relation_from_response({"s": node_data, "p": r["p"], "o": r["o"]}))
    #
    #     query2 = """PREFIX ns: <http://rdf.freebase.com/ns/>
    #                     SELECT DISTINCT * WHERE {
    #                                         ?s ?p """ + prefixed + """ . """ \
    #                                         + "FILTER(?s != " + prefixed + ")" \
    #                                         + self.__get_pred_filter__("?p", relations_to_filter=relations_to_filter) \
    #                                         + self.__get_entity_filter__("?s") \
    #                                         + "}"
    #     results2 = self.execute(query2)['results']['bindings']
    #     if results2 is not None and len(results2) > 0:
    #         for r in results2:
    #             triples.add(Relation.relation_from_response({"s": r["s"], "p": r["p"], "o": node_data}))
    #     return triples
    #
    # def two_steps(self, mid, strict=True, relations_to_filter=None):
    #     if mid[0:2] != 'm.':
    #         return []
    #     prefixed = "ns:" + mid
    #     node_data = {"type": "uri", "value": "http://rdf.freebase.com/ns/" + mid}
    #     query1 = """PREFIX ns: <http://rdf.freebase.com/ns/>
    #                             SELECT DISTINCT * WHERE {
    #                                            {
    #                                                """ + prefixed + """ ?p1 ?e2 .
    #                                                ?e2 ?p2 ?e3 .
    #                                                 FILTER(?e2 != ?e3 AND ?e3 != """ + prefixed + """ AND ?e2 != """ + prefixed + """) .""" \
    #                                                 + "FILTER NOT EXISTS {?e2 ns:type.object.name ?name} ." \
    #                                                  + self.__get_pred_filter__("?p1", relations_to_filter=relations_to_filter) \
    #                                                  + self.__get_pred_filter__("?p2", relations_to_filter=relations_to_filter) \
    #                                                  + self.__get_entity_filter__("?e2") \
    #                                                  + self.__get_entity_filter__("?e3") \
    #                                                  + """}
    #                             }
    #             """
    #     triples = set()
    #     if results1 is not None and len(results1) > 0:
    #         results1 = self.execute(query1)['results']['bindings']
    #         for r in results1:
    #             # triples.add(Relation.relation_from_response({"s": node_data, "p": r["p1"], "o": r["e2"]})) # these edges will be added in one-hop
    #             triples.add(Relation.relation_from_response({"s": r["e2"], "p": r["p2"], "o": r["e3"]}))
    #
    #     if strict:
    #         return triples
    #     query2 = """PREFIX ns: <http://rdf.freebase.com/ns/>
    #                                             SELECT DISTINCT * WHERE {
    #                                                            {
    #                                                                """ + prefixed + """ ?p1 ?e2.
    #                                                                ?e3 ?p2 ?e2 .
    #                                                                FILTER(?e2 != ?e3 AND ?e2 != """ + prefixed + """ AND ?e3 != """ + prefixed + """) .""" \
    #                                                                 + "FILTER NOT EXISTS {?e2 ns:type.object.name ?name} ." \
    #                                                                 + self.__get_pred_filter__("?p1", relations_to_filter=relations_to_filter) \
    #                                                                 + self.__get_pred_filter__("?p2", relations_to_filter=relations_to_filter) \
    #                                                                 + self.__get_entity_filter__("?e2") \
    #                                                                 + self.__get_entity_filter__("?e3") \
    #                                                            + """}
    #                                             }
    #                             """
    #
    #     results2 = self.execute(query2)['results']['bindings']
    #     if results2 is not None and len(results2) > 0:
    #         for r in results2:
    #             # triples.add(Relation.relation_from_response({"s": node_data, "p": r["p1"], "o": r["e2"]})) # these edges will be added in one-hop
    #             triples.add(Relation.relation_from_response({"s": r["e3"], "p": r["p2"], "o": r["e2"]}))
    #
    #     query3 = """PREFIX ns: <http://rdf.freebase.com/ns/>
    #                                     SELECT DISTINCT * WHERE {
    #                                                    {
    #                                                        ?e1 ?p1 """ + prefixed + """ .
    #                                                        ?e3 ?p2 ?e1 .
    #                                                        FILTER(?e1 != ?e3 AND ?e3 != """ + prefixed + """ AND ?e1 != """ + prefixed + """) .""" \
    #                                                          + "FILTER NOT EXISTS {?e1 ns:type.object.name ?name} ." \
    #                                                          + self.__get_pred_filter__("?p1", relations_to_filter=relations_to_filter) \
    #                                                          + self.__get_pred_filter__("?p2", relations_to_filter=relations_to_filter) \
    #                                                          + self.__get_entity_filter__("?e1") \
    #                                                          + self.__get_entity_filter__("?e3") \
    #                                                          + """}
    #                                     }
    #                     """
    #     results3 = self.execute(query3)['results']['bindings']
    #     if results3 is not None and len(results3) > 0:
    #         for r in results3:
    #             # triples.add(Relation.relation_from_response({"s": r["e1"], "p": r["p1"], "o": node_data})) # these edges will be added in one-hop
    #             triples.add(Relation.relation_from_response({"s": r["e3"], "p": r["p2"], "o": r["e1"]}))
    #
    #     query4 = """PREFIX ns: <http://rdf.freebase.com/ns/>
    #                                             SELECT DISTINCT * WHERE {
    #                                                            {
    #                                                                ?e1 ?p1 """ + prefixed + """ .
    #                                                                ?e1 ?p2 ?e3 .
    #                                                                FILTER(?e1 != ?e3 AND ?e3 != """ + prefixed + """ AND ?e1 != """ + prefixed + """) .""" \
    #                                                                  + "FILTER NOT EXISTS {?e1 ns:type.object.name ?name} ." \
    #                                                                 + self.__get_pred_filter__("?p1", relations_to_filter=relations_to_filter) \
    #                                                                 + self.__get_pred_filter__("?p2", relations_to_filter=relations_to_filter) \
    #                                                                 + self.__get_entity_filter__("?e1") \
    #                                                                 + self.__get_entity_filter__("?e3") \
    #                                                            + """}
    #                                             }
    #                             """
    #     results4 = self.execute(query4)['results']['bindings']
    #     if results4 is not None and len(results4) > 0:
    #         for r in results4:
    #             # triples.add(Relation.relation_from_response({"s": r["e1"], "p": r["p1"], "o": node_data})) # these edges will be added in one-hop
    #             triples.add(Relation.relation_from_response({"s": r["e1"], "p": r["p2"], "o": r["e3"]}))
    #     return triples

    def shortestPathResponse(self, entity_1, entity_2, threshold=2, relationsToFilter=None):
        '''
        :param entity_1: mid
        :param entity_2: mid
        :param threshold:
        :return: length of path, list of relation instances
        '''
        query_template = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT * WHERE {<CLAUSES>}"
        entity_1 = "ns:" + entity_1
        entity_2 = "ns:" + entity_2

        variable_set = ["?" + str(chr(i)) for i in range(97, 97 + threshold)] # 4 edges -> 3 nodes on the path
        predicate_set = ["?" + str(chr(i)) for i in range(122, 122 - threshold - 1, -1)] # 4 edges -> 4 predicates

        curr_len = 0
        while curr_len <= threshold:
            predicates = predicate_set[:curr_len + 1]

            entities1 = [entity_1] + variable_set[:curr_len] + [entity_2]
            clause_set1, clause1 = self.__queryClause__(entities1, predicates, relationsToFilter=relationsToFilter)
            query1 = query_template.replace("<CLAUSES>", clause1)
            result1 = self.execute(query1)[u'results'][u'bindings']
            if result1 is not None and len(result1) > 0:
                return curr_len, self.__relationsFromResponse__(clause_set1, result1, relationsToFilter=relationsToFilter)

            entities2 = [entity_2] + variable_set[:curr_len] + [entity_1]
            clause_set2, clause2 = self.__queryClause__(entities2, predicates, relationsToFilter=relationsToFilter)
            query2 = query_template.replace("<CLAUSES>", clause2)
            result2 = self.execute(query2)[u'results'][u'bindings']
            if result2 is not None and len(result2) > 0:
                return curr_len, self.__relationsFromResponse__(clause_set2, result2, relationsToFilter=relationsToFilter)
            curr_len +=1
        return -1, None

    def shortestPathLength(self, entity_1, entity_2, threshold=4):
        '''
        :param entity_1: mid
        :param entity_2: mid
        :param threshold: maximum path length
        :return: length of shortest path, in terms of edge (1-hop, 2-hop, etc)
                If no path exists smaller than threshold, then return -1
        '''
        length, relations = self.shortestPathResponse(entity_1, entity_2, threshold=threshold)
        return length

    def __queryClause__(self, entities, predicates, relationsToFilter=None):
        clause_set = set()
        for i in range(len(predicates)):
            clause_set.add((entities[i], predicates[i], entities[i + 1]))
        clause = ""
        for c in clause_set:
            predicate = c[1]
            clause += "\n" + c[0] + "\t" + predicate + "\t" + c[2] + " ."
            clause += "\n" + self.__get_pred_filter__(predicate, relations_to_filter=relationsToFilter)
        for i in range(len(entities) - 1):
            e1 = entities[i]
            e2 = entities[i+1]
            if e1.startswith("?") or e2.startswith("?"):
                clause += "\nFILTER(" + entities[i] + " != " + entities[i+1] + ")."
        for e in entities:
            if e.startswith("?"):
                clause += "\n" + self.__get_entity_filter__(e)
        return clause_set, clause

    def __relationsFromResponse__(self, clause_set, result, relationsToFilter=None):
        relations = []
        for response in result:
            for c in clause_set:
                entity_1 = c[0]
                predicate = c[1]
                entity_2 = c[2]
                relation_id = self.remove_uri_prefix(response[predicate[1:]]['value'])
                if relation_id in relationsToFilter:
                    continue
                if entity_1.startswith("?"):
                    source = Node.from_sparql(response[entity_1[1:]])
                else:
                    source = Entity(mid=entity_1.replace("ns:", ""))
                if entity_2.startswith("?"):
                    target = Node.from_sparql(response[entity_2[1:]])
                else:
                    target = Entity(mid=entity_2.replace("ns:", ""))

                relation = Relation(relation_id, source=source, target=target)
                relations.append(relation)
        return relations

    def relation_path_to_query(self, relation_path):
        if len(relation_path) == 2:
            var = "ns:" + relation_path[0] + " ?y .\n" + "?y ns:" + relation_path[1]
        else:
            var = "ns:" + relation_path[0]
        return var

    def relation2_path_to_query(self, relation_path):
        if len(relation_path) == 2:
            var = "ns:" + relation_path[0] + " ?z .\n" + "?z ns:" + relation_path[1]
        else:
            var = "ns:" + relation_path[0]
        return var

    def parse_rel_result(self, results, relations_to_filter=None):
        rel_list = []
        if results is None or len(results) == 0:
            return rel_list
        for r in results:
            p1 = self.remove_uri_prefix(r["p1"]["value"])
            p2 = None
            if "p2" in r: p2 = self.remove_uri_prefix(r["p2"]["value"])
            ct = None
            if "ct" in r: ct = r["ct"]["value"]
            if relations_to_filter is not None:
                if p1 not in relations_to_filter:
                    continue
                if p2 is not None and p2 not in relations_to_filter:
                    continue
            rel_list.append({"p1": p1, "p2": p2, "ct": ct})
        return rel_list

    def __get_filter__(self, varname, relations_to_filter):
        if relations_to_filter:
            filters = ["^http://rdf.freebase.com/ns/"+ r for r in relations_to_filter]
            filter = '|'.join(filters)
            return 'FILTER(STRSTARTS(STR(' + varname + '), "http://rdf.freebase.com/ns/") AND((REGEX(STR(' + varname + '), "' + filter +'")))) . \n'
        return ""

    def __get_pred_filter__(self, varname, relations_to_filter=None):
        return 'FILTER(STRSTARTS(STR(' + varname + '), "http://rdf.freebase.com/ns/") AND(!(REGEX(STR(' + varname + '), "^http://rdf.freebase.com/ns/type.|^http://rdf.freebase.com/ns/common.|^http://rdf.freebase.com/ns/base.descriptive_names|^http://rdf.freebase.com/ns/freebase.|^http://rdf.freebase.com/ns/base.ontologies")))) . \n'


    def __get_entity_filter__(self, varname):
        return 'FILTER((STRSTARTS(STR(' + varname + '), "http://rdf.freebase.com/ns/m."))' + \
                'OR(isLiteral(' + varname + ')' +  \
                'AND(langMatches(lang(' + varname + '), "en")' + \
                'OR lang(' + varname + ') = "en"))) . \n'

    def remove_uri_prefix(self, entry):
        idx = entry.find('ns/')
        if idx != -1:
            return entry[idx + 3:]
        return entry

if __name__ == '__main__':
    class Constraint(object):

        def __init__(self, mid, name, relation, is_ans_constraint, surface_form, start_index, end_index):
            self.mid = mid
            self.name = name
            self.relation = relation
            self.is_ans_constraint = is_ans_constraint
            self.surface_form = surface_form
            self.start_index = start_index
            self.end_index = end_index

        def __str__(self):
            return "constraint: " + str(self.mid) + " " + str(self.name) + " " + str(self.relation) + " " + str(
                self.is_ans_constraint)

    sparql = sparqlUtils()
    start = time.time()
    #jb = 'm.06w2sn5'
    #jax = 'm.0gxnnwq'
    #sparql.shortestPathResponse(jb, jax)

    # print(sparql.one_hop_expansion(mid='m.0gxnnwq'))
    # print(sparql.one_hop_rev_expansion(mid='m.0gxnnwq'))
    # print(sparql.two_hop_expansion(mid='m.02639ym'))
    # print(sparql.two_hop_rev_expansion(mid='m.02639ym'))
    # print(sparql.eval_one_hop_expansion('m.0gxnnwq', 'people.person.gender'))
    # print(sparql.eval_one_hop_rev_expansion('m.0gxnnwq', 'location.location.people_born_here'))
    # print(sparql.eval_two_hop_expansion('m.02639ym', 'business.employer.employees', 'business.employment_tenure.title'))
    # print(sparql.eval_two_hop_rev_expansion('m.02639ym', 'government.governmental_jurisdiction.governing_officials', 'government.government_position_held.governmental_body'))
    # print(sparql.get_interim_size('m.02639ym',['government.governmental_jurisdiction.governing_officials', 'government.government_position_held.governmental_body'], True))
    # relations = {'relations': ['government.governmental_jurisdiction.governing_officials', 'government.government_position_held.governmental_body'], 'is_reverse': True}
    # print(sparql.get_connecting_path('m.02639ym', relations, "m.0653m"))
    # print(sparql.evaluate_connecting_path('m.02639ym', relations, {'is_reverse': False, 'relations': ['location.country.official_language']}))
    # print(sparql.get_all_cvt_constraints('m.02639ym', ['government.governmental_jurisdiction.governing_officials', 'government.government_position_held.governmental_body'], True, "m.04g702_"))
    #
    # c1= Constraint("m.04g702_", "", "government.government_position_held.office_holder", False,"",0,4)
    # c2 = Constraint("m.01mp", "", "common.topic.notable_types", True, "",1,2)
    # print(sparql.eval_all_constraints('m.02639ym', ['government.governmental_jurisdiction.governing_officials', 'government.government_position_held.governmental_body'], [c1,c2], True))
    # #
    # ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]
    # #
    # print(sparql.get_ans_constraint_candidates('m.02639ym', ['government.governmental_jurisdiction.governing_officials', 'government.government_position_held.governmental_body'], ANS_CONSTRAINT_RELATIONS, True))
    # print(sparql.get_ans_constraint_rel('m.02639ym', ['government.governmental_jurisdiction.governing_officials', 'government.government_position_held.governmental_body'], 'm.01mp', relations_to_filter=None, constraint_relations_to_filter=ANS_CONSTRAINT_RELATIONS, is_reverse=True))

    #print(sparql.eval_one_hop_rev_expansion('m.01914', 'location.country.capital'))
    # print(sparql.eval_two_hop_rev_expansion('m.01914', 'olympics.olympic_bidding_city.olympics_bid_on',
    #                                         'olympics.olympic_city_bid.bidding_city'))

    #print(sparql.get_conjunction_path("m.0jzc", {"relations": ['language.human_language.countries_spoken_in'], "is_reverse": False}, "\"1973\"^^xsd:dateTime"))

    print(sparql.evaluate_conjunction_path("m.0jzc",{"relations": ['language.human_language.countries_spoken_in'], "is_reverse": False},
                                           {'is_reverse': False,
                                            'relations': ["location.statistical_region.life_expectancy",
                                                          "measurement_unit.dated_float.date"]},
                                           "\"1973\"^^xsd:dateTime"))

    print(sparql.evaluate_conjunction_path("m.0jzc", {"relations": ['language.human_language.countries_spoken_in'],
                                                      "is_reverse": False},
                                           {'is_reverse': False,
                                            'relations': ["religion.religious_leadership_jurisdiction.leader",
                                                          "religion.religious_organization_leadership.start_date"]},
                                           "\"1973\"^^xsd:dateTime"))