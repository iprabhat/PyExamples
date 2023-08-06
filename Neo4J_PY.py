# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:01:33 2018

@author: prabhat
"""

from neo4j.v1 import GraphDatabase

class HelloWorldExample(object):
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()


    @staticmethod
    def _read_data(tx, query):
        result = tx.run(query)        
        return result

    def print_data(self, query):
        with self._driver.session() as session:
            #Currently not used
            #greeting = session.write_transaction(self._read_data, query)
            with session.begin_transaction() as tx:
                result = tx.run(query)
                for r in result:
                    print(r)
            #for r in greeting:
            #    print(r)
        


def getData():
    A = HelloWorldExample("bolt://localhost:7687","neo4j","admin")
    A.print_data("match(a:Person)-[:knows]->(b) return a,b")


if(__name__ =='__main__'):
    getData()