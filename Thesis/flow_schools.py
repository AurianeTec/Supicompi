if __name__ == '__main__':
    import pandas as pd
    import os
    os.environ['USE_PYGEOS'] = '0'
    import geopandas as gpd
    pd.options.mode.chained_assignment = None  # default='warn'
    import networkx as nx
    import shapely
    import multiprocess as mp
    import numpy as np
    import igraph as ig
    from ta_lab.assignment.assign import frank_wolfe
    from ta_lab.assignment.line import *
    from ta_lab.assignment.graph import *
    from ta_lab.assignment.shortest_path import ShortestPath as SPP
    from ast import literal_eval

    #### FUNCTION IMPORTS ####
    
    #--- Custom function (Anastassia)
    # Create a dictionary of attributes (useful for networkX)
    def make_attr_dict(*args, **kwargs): 
        
        argCount = len(kwargs)
        
        if argCount > 0:
            attributes = {}
            for kwarg in kwargs:
                attributes[kwarg] = kwargs.get(kwarg, None)
            return attributes
        else:
            return None # (if no attributes are given)
            


    #### CREATE NETWORK IN NETWORKX AND IGRAPH ####
    
    #--- Create the network in NetworkX
    # Retrieve edges
    edges_with_id = pd.read_csv('data/clean/initial_network_edges.csv')
    edges_with_id["geometry"] = edges_with_id.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    edges_with_id = gpd.GeoDataFrame(edges_with_id, geometry = 'geometry', crs = 4326).to_crs(2154)
    edges_with_id = edges_with_id.rename(columns={"id": "G"})

    # Retrieve nodes
    nodes_carbike_centroids_RER_complete = pd.read_csv('data/clean/initial_network_nodes_complete.csv')
    nodes_carbike_centroids_RER_complete["geometry"] = nodes_carbike_centroids_RER_complete.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    nodes_carbike_centroids_RER_complete = gpd.GeoDataFrame(nodes_carbike_centroids_RER_complete, geometry = 'geometry', crs = 2154)

    # Create the attr_dict
    nodes_carbike_centroids_RER_complete["attr_dict"] = nodes_carbike_centroids_RER_complete.apply(lambda x: make_attr_dict(
                                                                    nodetype = x.nodetype,
                                                                    centroid = x.centroid,
                                                                    RER = x.RER,
                                                                    IRIS = x.CODE_IRIS,
                                                                    pop_dens = x.pop_density,
                                                                    active_pop_density = x.active_pop_density,
                                                                    school_pop_density = x.school_pop_density,
                                                                    num_schools = x.school_count,
                                                                    num_jobs = x.num_jobs,
                                                                    ),
                                                                    axis = 1) 

    # Create Graph with all nodes and edges
    G = nx.from_pandas_edgelist(edges_with_id, source='x', target='y', edge_attr=True)
    G.add_nodes_from(nodes_carbike_centroids_RER_complete.loc[:,["osmid", "attr_dict"]].itertuples(index = False))

    #--- Moving from NetworkX to igraph
    g_igraph = ig.Graph()
    networkx_graph = G
    g_igraph = ig.Graph.from_networkx(networkx_graph)

    # eids: "conversion table" for edge ids from igraph to nx 
    eids_nx = [g_igraph.es[i]["G"] for i in range(len(g_igraph.es))]
    eids_ig = [i for i in range(len(g_igraph.es))]
    eids_conv = pd.DataFrame({"nx": eids_nx, "ig": eids_ig})    

    # nids: "conversion table" for node ids from igraph to nx
    nids_nx = [g_igraph.vs(i)["_nx_name"][0] for i in range(len(g_igraph.vs))]
    nids_ig = [i for i in range(len(g_igraph.vs))]
    nids_conv = pd.DataFrame({"nx": nids_nx, "ig": nids_ig})
    nids_conv['nx'] = nids_conv['nx'].astype(int)

    # combine the conversion table with nodes_carbike_centroids_RER_complete
    nodes_carbike_centroids_RER_complete = nodes_carbike_centroids_RER_complete.merge(nids_conv, left_on = "osmid", right_on = "nx", how = "left")
    nodes_carbike_centroids_RER_complete = nodes_carbike_centroids_RER_complete.drop(columns = ["nx"])

    # Isolate centroids
    from itertools import combinations
    seq = g_igraph.vs.select(centroid_eq = True)
    centroids = [v.index for v in seq]
    # centroids = centroids[0:3] #TODO


    #### SETTING THE NUMBER OF CPUs ####
    num_processes = 30
    baseline_schoolpopdens = pd.read_csv('data/results/baseline_schoolpopdens.csv')



    #### TRAVEL ASSIGNMENT ####
    print('starting travel assignment')
    #--- Create network compatible with frank_wolfe function
    nt = Network('net')
    node = Vertex("a")

    with open("data/clean/network.csv") as fo: 
        lines = fo.readlines()[1:]
        for ln in lines:
            eg = ln.split(',')
            nt.add_edge(Edge(eg))
    nt.init_cost()       
    g_df = pd.read_csv("data/clean/network.csv") 

    # create dictionary of igraph ID to modified osmID
    centroid_igraph_to_mod_osmID = {}
    for i in range(len(centroids)):
        centroid_igraph_to_mod_osmID[i] = nodes_carbike_centroids_RER_complete.loc[nodes_carbike_centroids_RER_complete['ig'] == centroids[i]]['osmid'].apply(lambda x: 'N'+ (str(x) + '.0').zfill(5)).values[0]


    #--- Run frank-wolfe
    print('starting Frank Wolfe')
    vol2 = None

    # Get OD matrix
    OD = baseline_schoolpopdens
    OD.columns = OD.columns.astype(int)

    # Rename the columns and rows according to the modified osmID 
    OD = OD.rename(columns = {i : centroid_igraph_to_mod_osmID[i] for i in range(len(OD))}) #rename index of centroid as osmid of centroid
    OD.index = OD.columns

    # From all centroids to all centroids
    origins = OD.columns
    destinations = origins
    
    vol2 = frank_wolfe(nt, OD, origins, destinations)
    # dicts.append(vol2)
    vol2_df = pd.DataFrame.from_dict(vol2, orient='index')
    vol2_df.to_csv('data/results/traffic_flow_baselineschoolpopdens.csv')
    

    #### CALCULATE BENEFIT METRIC ####
    
    print('starting benefit metric')
    # Define the file path
    file_path = './data/clean/identified_gaps_under80.csv'  
    mygaps = pd.read_csv(file_path, chunksize=100000) 

    def process_row(row, edge_lengths, g_df): #for one csv
        try:
            row['path'] = literal_eval(row['path'])
            row["B_star_baseline"] = np.sum([vol2[g_df.loc[g_df['G'] == eids_conv_dict[i]]['edge'].values[0]] * edge_lengths[i] for i in row.path])
            row["B_baseline"] = row["B_star_baseline"] / row["length"]
            return row
        except Exception as e:
            print("Exception occurred at index:", row.name)
            print("Exception message:", str(e))
            return None

    def process_chunk(chunk): #for one csv
        processed_rows = [process_row(row, edge_lengths, g_df) for _, row in chunk.iterrows()]
        processed_chunk = pd.DataFrame(processed_rows)
        return processed_chunk


    eids_conv_dict = eids_conv.set_index('ig')['nx'].to_dict()
    edge_lengths = {i: g_igraph.es[i]["length"] for i in range(len(g_igraph.es))}
    pool = mp.Pool(processes=num_processes)

    results = pool.map(process_chunk, mygaps)
    pool.close()
    pool.join()


    #### SAVE RESULTS ####
    # Open the output file in append mode
    output_file = "./data/clean/gaps_benefit_metric_schools_baseline.csv"
    with open(output_file, "a") as f:
            for df_chunk in results:
                    df_chunk.to_csv(f, header=f.tell() == 0, index=False)
