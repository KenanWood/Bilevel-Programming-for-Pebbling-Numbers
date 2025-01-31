#!/usr/bin/env python3
# import docplex.mp.model as cpx      # The bilevel linear programming
# import cplex                        # API for modeling the program as a bilevel linear program
import networkx as nx               # Software for large network visualization
import re
from time import time
from itertools import combinations
from subprocess import Popen, PIPE
from sage.all import *
import gurobipy as gp
import ast

from random import sample

DIRECTORY = "/home/DAVIDSON/kewood/Pebbling" # Replace with current directory

def get_lemke_graph(index: int = 0) -> nx.Graph:
    L = nx.Graph()
    if index == 0:
        L.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,1), (3,6), (3,8), (4,7), (5,7), (5,8), (7,8)])
    elif index == 1:
        L.add_edges_from([(1,2), (1,6), (2,3), (2,6), (3,4), (3,7), (4,5), (4,7), (4,8), (5,6), (6,7), (6,8)])
    elif index == 2:
        L.add_edges_from([(1,2), (1,6), (2,3), (3,4), (3,7), (3,5), (4,5), (4,7), (4,8), (5,6), (5,8), (6,7), (6,8), (7,8)])
    else:
        raise Exception("Index out of range. Index should be an int in {0, 1, 2}")
    return L

def get_lemke_squared(index: int = 0) -> nx.Graph:
    L = get_lemke_graph(index)
    return nx.cartesian_product(L, L)

def get_nonisomorphic_roots(G: nx.Graph) -> list:
    return [orbit[0] for orbit in Graph(G).automorphism_group(orbits=True)[1]]

def get_bruhat_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (1, 5),
        (2, 4), (2, 8),
        (3, 4), (3, 21),
        (4, 24),
        (5, 6), (5, 9),
        (6, 7), (6, 10),
        (7, 8), (7, 11),
        (8, 12),
        (9, 10), (9, 17),
        (10, 13), 
        (11, 12), (11, 14),
        (12, 20),
        (13, 14), (13, 15),
        (14, 16), (15, 16), (15, 18),
        (16, 19), (17, 18), (17, 21),
        (18, 22), (19, 20), (19, 23),
        (20, 24), (21, 22),
        (22, 23), (23, 24)
    ])
    return G

'''
NOTE: instead of looping over all k-subsets of V(G) for the support, loop over the "nonisomorphic" ones.
We can get these ones by constructing a set of "isomorphism" classes of the k-sets.

Algorithm:
Initialize a set of k-subsets of V(G), all nonisomorphic, called non_iso_sets := {}
Initialize a set of k-subsets of V(G), call it all_sets := {}
For all k-subsets S of V(G) such that r not in S:
    If S in all_sets: continue
    # Else do the following:
    Add S to non_iso_sets
    Initialize set iso_class := {}, the isomorphism class of S
    For phi in Aut(G) that fixes r:
        Add phi(S) to iso_class
    Add the contents of iso_class to P
Return non_iso_sets
'''
def get_nonisomorphic_sets(G: nx.Graph, r, k: int):
    Aut = Graph(G).automorphism_group()
    non_iso_sets = set()
    checked_sets = set()
    for S in combinations(G.nodes, k):
        if r in S: continue

        set_S = frozenset(S)
        if set_S in checked_sets: continue

        non_iso_sets.add(set_S)
        for phi in Aut:
            if phi(r) != r: continue # require r is fixed under phi
            phi_S = frozenset([phi(x) for x in S])
            checked_sets.add(phi_S)
    
    return non_iso_sets


def generate_mps_aux(G: nx.Graph, root, support: frozenset, filename: str):
    diG = G.to_directed()
    V = list(diG.nodes())
    A = list(diG.edges())
    S = list(support)
    diam = nx.diameter(G)

    if root not in V:
        raise Exception("Specified root not a vertex of graph")
    if not support.issubset(set(V)):
        raise Exception("Support not a set of vertices")
    
    model = gp.Model(name="MIP")

    f = open(filename+".aux", "w")
    f.write('N %i \n' %len(A)) # number of variables = |A|
    f.write('M %i \n' %(len(V)+1)) # number of constraints = |V|+1

    # Map from vertices of graph to variable
    y = {}
    z = {}

    ### LOWER LEVEL VARIABLES
    for i in range(len(A)):
        z[A[i]] = model.addVar(name='z'+str(i), vtype=gp.GRB.INTEGER, ub=2**(diam-1))
        f.write('LC %i \n' %i) # z[i] is a lower level variable. IMPORTANT: must add z variables first

    ### UPPER LEVEL VARIABLES
    for i in range(len(S)):
       y[S[i]] = model.addVar(name='y'+str(i), vtype=gp.GRB.INTEGER, ub=2**diam)
    for v in V:
        if v not in S:
            y[v] = 0

    ### LOWER LEVEL CONSTRAINTS
    model.addConstr(gp.quicksum( z[(root,u)] for u in V if (root,u) in A ) <= 0, name="ZeroOut")
    f.write('LR 0\n') # 1 not 0 because the objective will be printed first. TEST THIS

    for i in range(len(V)):
        v = V[i]
        model.addConstr(2*gp.quicksum(z[(v,u)] for u in V if (v,u) in A) - 
                        gp.quicksum(z[(u,v)] for u in V if (u,v) in A) <= y[v], name="Flow"+str(i))
        f.write("LR %i\n" %(i+1))

    
    
    ### LOWER LEVEL OBJECTIVE
    for i in range(len(A)):
        coef = 0 # coefficient on z[(A[i])]
        if A[i][1] == root: # If the head of the edge is the root; equivalent to A[i] in delta^-(r)
            coef = 1
        f.write('LO %i \n' % coef)

    ### LOWER LEVEL OBJECTIVE SIGN
    f.write('OS -1') # maximization
    f.close()

    ### UPPER LEVEL CONSTRAINTS
    model.addConstr(gp.quicksum(z[(u,root)] for u in V if (u,root) in A) <= 0, name="UnSol")
    model.addConstr(gp.quicksum(y[v] for v in S) >= max(len(V), 2**diam)) # Should be infeasible if Class 0

    ### UPPER LEVEL OBJECTIVE
    model.setObjective(gp.quicksum(-y[v] for v in S), gp.GRB.MINIMIZE)

    ### WRITE THE FILE TO MPS
    model.write(filename+".mps")

    model.write(filename+".lp")


def parse_output(output):
    pattern = re.compile(r"LEADER COST\s([-.\d]+) .*FOLLOWER COST ([-.\d]+)", re.MULTILINE)
    results = re.finditer(pattern, str(output))
    good = re.search("OK, solved to optimality", output)
    bad = re.search("WARNING", output)
    bound = re.search("No solution exists", output)
    if good is not None and bad is None:
        pass
    elif bound is not None:
        print("OK, bound is too high!")
        return None
    else:
        # print(output)
        print("Warning!")
        return None

    for result in results:
        return float(result.group(1))


def parse_output_infeasible(output):
    optimal = re.search("OK, solved to optimality", output)
    warning = re.search("WARNING", output)
    feasibility_unknown = re.search("No solution exists", output)
    infeasible = re.search("Master MIP not solved, cplex status 103 --> Model is infeasible", output)
    # bound = re.search("No solution exists", output)
    if optimal is not None and warning is None:
        pattern = re.compile(r"LEADER COST\s([-.\d]+) .*FOLLOWER COST ([-.\d]+)", re.MULTILINE)
        results = re.finditer(pattern, str(output))
        for result in results:
            return float(result.group(1)) # get optimal value of leader
        
    if infeasible is not None and warning is None:
        return "Infeasible"

    if feasibility_unknown is not None and not infeasible is None:
        print("Failed to solve.")
        return None

    if warning is not None:
        print("Warning!")
        return None

def rooted_pi_S(G: nx.Graph, root, support: frozenset, filename: str, 
                dump: bool = False, dump_filename: str = "", non_class0_filename: str = ""):
    '''
    Dump the output contents into dump_filename if dump == True and into 
    non_class0_filename if the pebbling number is greater than |G|.
    '''
    if root not in G.nodes: raise Exception("The root is not a vertex of the graph")

    generate_mps_aux(G, root, support, filename)

    process = Popen([
        "/opt/bilevel/bilevel", ########### REPLACE WITH BILEVEL DIRECTORY
        "-mpsfile", DIRECTORY + "/" + filename + ".mps", 
        "-auxfile", DIRECTORY + "/" + filename + ".aux", 
        "-time_limit", "30",
        "-available_memory","120000",
        "-print_sol", "2",
        "-setting","2"], 
        stdout=PIPE, 
        cwd="/opt/bilevel", 
        universal_newlines=True)
    
    (output, errors) = process.communicate()
    exit_code = process.wait()
    parsed_out = parse_output_infeasible(str(output)) # Replaced with the infeasibility approach
    if parsed_out is None:
        val = -1
        val_str = "undetermined"
    elif parsed_out == "Infeasible":
        val = max(len(G.nodes()), 2**(nx.diameter(G)))
        val_str = "p <= " + str(val) # the output file will look like pi_S(G, r) = p <= max(|G|, 2^(diam(G)))
    else: # We got some optimal solution
        val = -(parsed_out) + 1
        val_str = str(val)

    # Case when we've proven that (G, r) is not a class 0 rooted graph.
    if dump and val > len(G.nodes()) and len(non_class0_filename) > 0:
        with open(non_class0_filename, "a") as file:
            file.write("==============================================\n")
            file.write("INSTANCE\n")
            file.write("Graph:" + str((G.edges())) + "\n")
            file.write("Support: " + str(list(support)) + "\n")
            file.write("Root: " + str(root) + "\n")
            file.write("MPS file:\n")
            with open(filename+'.mps', 'r') as mpsfile:
                mps = mpsfile.read()
            file.write(mps + "\n")
            file.write("AUX file:\n")
            with open(filename + '.aux', 'r') as auxfile:
                aux = auxfile.read()
            file.write(aux + "\n")
            file.write("\n")
            file.write("==============================================\n")
            file.write("OUTPUT\n")
            file.write(output)
            file.write("pi_S(G, r) = " + val_str + "\n\n")
    
    # Any case
    if dump and len(dump_filename) > 0:
        with open(dump_filename, "a") as file:
            file.write("==============================================\n")
            file.write("INSTANCE\n")
            graph_string = "Graph:" + str((G.edges())) + "\n" # the string representation of the graph to write
            for i in range(3): # number of minimal Lemke graphs.
                for j in range(3):
                    if i == j and set(get_lemke_squared(i).edges()) == set(G.edges()):
                        graph_string = "Graph: Lemke squared " + str(i) + "\n"
                        break
                    if i != j and set(nx.cartesian_product(get_lemke_graph(i), get_lemke_graph(j)).edges()) == set(G.edges()):
                        graph_string = "Graph: Lemke" + str(i) + " x Lemke" + str(j) + "\n"
                        break
            file.write(graph_string)
            file.write("Support: " + str(list(support)) + "\n")
            file.write("Root: " + str(root) + "\n")
            file.write("pi_S(G, r) = " + val_str + "\n\n")
    
    return val


### NOT USED ANYMORE. WE USE THE COVERING METHOD.
def rooted_pi_k(G: nx.Graph, root, k: int, filename: str, 
                dump: bool = False, dump_filename: str = "", non_class0_filename: str = ""):
    non_iso_sets = get_nonisomorphic_sets(G, root, k)
    max_pi = 0

    for S in non_iso_sets:
        next_pi = rooted_pi_S(G, root, S, filename, dump, dump_filename, non_class0_filename) # dumps contents into file if needed
        max_pi = max(max_pi, next_pi)
    
    return max_pi


def pi_k(G: nx.Graph, k: int, filename: str, 
         dump: bool = False, dump_filename: str = "", non_class0_filename: str = ""):
    pi_num = 0
    for root in get_nonisomorphic_roots(G):
        next_pi = rooted_pi_k(G, root, k, filename, dump, dump_filename, non_class0_filename)
        pi_num = max(pi_num, next_pi)
    print("%i-pebbling number of" %k)
    print((G.nodes(), G.edges()))
    print("is equal to", pi_num)
    return pi_num



def num_k_cover(G: nx.Graph, k: int, cover_size: int):
    num_cover_sets = 0
    num_dict = {}
    for root in get_nonisomorphic_roots(G):
        non_iso_sets = get_nonisomorphic_sets(G, root, k)
        num_root = 0
        while len(non_iso_sets) > 0:
            cover_set = set()
            for S in non_iso_sets:
                if len(cover_set.union(S)) <= cover_size:
                    cover_set = cover_set.union(S)
            to_remove = set()
            for S in non_iso_sets:
                if S.issubset(cover_set):
                    to_remove.add(S)
            non_iso_sets = non_iso_sets.difference(to_remove)
            num_cover_sets += 1
            num_root += 1
        num_dict[root] = num_root
        # print(num_cover_sets)
    return num_dict

'''
Main method for our computations. If you want to check only a fixed set of roots (e.g. just one), then set roots
to a list of desired roots. 'roots' also specifies order.
'''
def pi_k_cover(G: nx.Graph, k: int, cover_size: int, filename: str,
               dump: bool = False, dump_filename: str = "", non_class0_filename: str = "",
               continue_computation: bool = True, roots: list = None):

    start_root = None
    if continue_computation:
        try:
            with open(dump_filename, "r") as file:
                contents = file.read()
                index = contents.rfind("Root:")
                start_root = ast.literal_eval(contents[index + 6: contents.rfind("\npi")])
        except:
            raise Exception("Could not open file or file was not properly formatted.")
        
    if roots is None:
        roots = get_nonisomorphic_roots(G)
    
    num_pi = 0
    begin_root = False
    start_time = time()
    curr_time = time()
    for root in roots:

        # Go until we hit the specified root if we are continuing
        if continue_computation:
            if not begin_root or root == start_root:
                begin_root = True
                curr_time = time()
            else:
                continue

        non_iso_sets = get_nonisomorphic_sets(G, root, k) # Restarts at the right place

        while len(non_iso_sets) > 0:
            cover_set = set()
            for S in non_iso_sets:
                if len(cover_set.union(S)) <= cover_size:
                    cover_set = cover_set.union(S)
            to_remove = set()
            for S in non_iso_sets:
                if S.issubset(cover_set):
                    to_remove.add(S)
            non_iso_sets = non_iso_sets.difference(to_remove)

            num_pi = max(num_pi, rooted_pi_S(G, root, cover_set, filename, dump, dump_filename, non_class0_filename))

        
        print("Time for root " + str(root) + ":", time() - curr_time)
        if dump:
            with open(dump_filename, "a") as file:
                file.write("Time for root " + str(root) + ": " + str(time() - curr_time) + "\n\n")
        curr_time = time()


    print("%i-pebbling number of" %k)
    print((G.nodes(), G.edges()))
    print("is equal to", num_pi)

    print("Total time:", time() - start_time)

    return num_pi

### Naive but simple way
def execute_parallel(G: nx.Graph, k: int, cover_size: int, id: int, num_machines: int):
    '''
    id: an int in [0, num_machines).
    '''
    roots = get_nonisomorphic_roots(G)

    roots_id = [roots[i] for i in range(len(roots)) if i % num_machines == id]

    # print("Computing on roots:", roots_id)

    graph_string = "Graph"
    for i in range(3): # number of minimal Lemke graphs.
        for j in range(3):
            if i == j and set(get_lemke_squared(i).edges()) == set(G.edges()):
                graph_string = "Lemke"+str(i)
                break
            if i != j and set(nx.cartesian_product(get_lemke_graph(i), get_lemke_graph(j)).edges()) == set(G.edges()):
                graph_string = "Lemke"+str(i)+"-x-Lemke"+str(j)
                break

    filename = "pi_"+str(k)+"_"+str(cover_size)+"_"+graph_string+"_id_"+str(id)
    dump = True
    dump_filename = "All_"+str(k)+"_"+str(cover_size)+"_"+graph_string+"_id_"+str(id)+".txt"
    non_class0_filename = "Non_Class0_"+str(k)+"_"+str(cover_size)+"_"+graph_string+"_id_"+str(id)+".txt"
    continue_computation = False

    pi_k_cover(G, k, cover_size, filename, dump, dump_filename, non_class0_filename, continue_computation, roots_id)



def pi(G: nx.Graph, filename: str, dump: bool = False, dump_filename: str = "", non_class0_filename: str = ""):
    return pi_k_cover(G, len(G.nodes()) - 1, len(G.nodes()) - 1, filename, dump, dump_filename, non_class0_filename, False)


if __name__ == "__main__":
    ####################################
    id = 0 # CHANGE TO ID OF MACHINE #
    ####################################
    num_machines = 7 ### UPDATE IF USING A DIFFERENT NUMBER OF MACHINES
    for i in range(3):
        for j in range(3):
            if i >= j: continue
            G = nx.cartesian_product(get_lemke_graph(i), get_lemke_graph(j))
            k = 4
            cover_size = 8
            execute_parallel(G, k, cover_size, id, num_machines)


    ### Testing failed cases

    # G = nx.cartesian_product(get_lemke_graph(0), get_lemke_graph(2))
    # root = (5,4)
    # support = frozenset([(2, 4), (2, 1), (4, 6), (2, 6), (2, 2), (6, 3), (4, 1), (2, 8)])
    # filename = "failed_cases"
    # dump = True
    # dump_filename = "Failed_cases.txt"
    # non_class0_filename = "Failed_cases_counterexamples.txt"
    # rooted_pi_S(G, root, support, filename, dump, dump_filename, non_class0_filename)


    ### Testing hypercubes

    # k = 2
    # while True:
    #     Q = nx.hypercube_graph(k)
    #     filename = "hypercube"
    #     dump_filename = "hypercube.txt"
    #     non_class0_filename = "Non-class0-hypercube.txt"
    #     dump = True
    #     pi(Q, filename, dump, dump_filename, non_class0_filename)
    #     k += 1

    ### Testing the Bruhat graph

    # B = get_bruhat_graph()
    # filename = "bruhat"
    # dump_filename = "All_bruhat.txt"
    # non_class0_filename = "Non-class0-bruhat.txt"
    # dump = True
    # pi(B, filename, dump, dump_filename, non_class0_filename)


