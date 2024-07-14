import networkx as nx


# Dictionary for the correspondeces between ICC codes and actual country names, 
# To get the codes/names information, refer to DOCUMENTS/EGM_2019_UserGuide.pdf (Annex C)
ICC_labels = {'GE':'Georgia',
            'IT': 'Italy',
            "FR": "France",
            "LT": "Lithuania",
            "SK": "Slovakia",
            "PT": "Portugal",
            "ES": "Spain",
            "EE": "Estonia",
            "RS": "Serbia",
            "HU": "Hungary",
            "IE": "Ireland",
            "PL": "Poland",
            "AT": "Austria",
            "LU": "Luxemburg",
            "CH": "Switzerland",
            "BE": "Belgium",
            "CZ": "Czech Republich",
            "NL": "Netherlands",
            "GR": "Greece",
            "GB": "Great Britain",
            "RO": "Romania",
            "ND": "Northen Ireland",
            "MD": "Moldova",
            "DK": "Denmark",
            "NO": "Norwey",
            "SE": "Sweden",
            "FI": "Finland",
            "LV": "Latvia",
            "DE": "Germany",
            "SI": "Slovenia",
            "HR": "Croatia",
            "BG": "Bulgaria",
            "UA": "Ukraine",
            "MK": "Macedonia"}



def add_node(G, coords, node_coords, node_counter):
    """"
    if node is already present, it returns old node index and the current node index
    es: node_counter is 64, node is already present with index 45, then the function returns the pair  (45, 64))
    if node was not present, it assigns current node_counter to the node, adds it to graph and increases current node_counter
    es: node_counter was 64, node is not present, then it returns (64, 65)
    """
    if coords not in node_coords:
        node_coords[coords] = node_counter # add a new entry to dictionary
        G.add_node(node_counter, longitude= coords[0], latitude= coords[1]) # increments node index to get ready for next node
        node_counter += 1
    return node_coords[coords], node_counter


def flexible_matching(key, coord, threshold=0.01):
    # Check if the absolute difference between the coordinate and the key is within the threshold
    if abs(key - coord) < threshold:
        return True
    else:
        return False

# boolean function to check for edge's presence
def check_edge(G, node_from, node_to):
    # Check if there's a path between 'node_from' and 'node_to'
    if nx.has_path(G, node_from, node_to):
        # Get the shortest path between 'node_from' and 'node_to'
        all_paths = list(nx.all_shortest_paths(G, node_from, node_to))
        #print("paths number= ", len(all_paths))
        for path in all_paths:
            # Get the labels of all nodes in the path
            path_labels = [G.nodes[node]['label'] for node in path]
            path_is_near_city = [G.nodes[node]['is_near_city'] for node in path]
            node_from_label = G.nodes[node_from]['label']
            node_to_label = G.nodes[node_to]['label']
            if all(label == 'empty' for label in path_labels[1:-1]) and all(label in {'empty', node_from_label, node_to_label} for label in path_is_near_city):
                return True
            else: 
                return False
    else: 
        return False

# Function to find a node by its label
def find_node_by_label(G, label):
    for node, data in G.nodes(data=True):
        if data.get('label') == label:
            return node
    return None