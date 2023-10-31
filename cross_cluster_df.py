import pandas as pd
import networkx as nx

def cross_cluster(dataframe: pd.DataFrame, column_a: str, column_b: str, additional_columns: list) -> pd.DataFrame:
    """
    Create clusters based on the connections between two types of entities in a given DataFrame.
    Each name in optional argument additional_columns should have a 1-to-1 relationship with column_b.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame containing the relationship data.
        column_a (str): The name of the first column in the DataFrame representing the first entity.
        column_b (str): The name of the second column in the DataFrame representing the second entity.
        additional_columns (list): List of names of additional columns that need to be aggregated. 

    Returns:
        pd.DataFrame: A DataFrame where each row represents a cluster with two lists of entity IDs from each column.

    Raises:
        ValueError: If the specified columns are not found in the DataFrame.

    Example:
        >>> df = pd.DataFrame({'employeeId': ['E1', 'E2', 'E2'], 'departmentId': ['D5', 'D5', 'D6']})
        >>> create_clusters(df, 'employeeId', 'departmentId', [])
    """
    if column_a not in dataframe.columns or column_b not in dataframe.columns or any(col not in dataframe.columns for col in additional_columns):
        raise ValueError("One or more specified columns not found in the DataFrame")

    G = nx.Graph()
    additional_info = {col: {} for col in additional_columns}

    for _, row in dataframe.iterrows():
        entity_a_id = str(row[column_a])
        entity_b_id = str(row[column_b])

        entity_a_node = f'a{entity_a_id}'
        entity_b_node = f'b{entity_b_id}'
        G.add_edge(entity_a_node, entity_b_node)

        # Collect data from additional columns
        for col in additional_columns:
            additional_info[col][entity_b_id] = row[col]

    connected_components = list(nx.connected_components(G))
    clusters = []

    for component in connected_components:
        entities_a = {n[1:] for n in component if n.startswith('a')}
        entities_b = {n[1:] for n in component if n.startswith('b')}

        cluster = {
            f'{column_a}s': list(entities_a),
            f'{column_b}s': list(entities_b)
        }

        # Aggregate additional column data
        for col in additional_columns:
            col_data = [additional_info[col][entity] for entity in entities_b if entity in additional_info[col]]
            cluster[f'{column_b}_{col}'] = col_data

        clusters.append(cluster)

    return pd.DataFrame(clusters)

# Sample data
data = {
    'employeeId': ['E1', 'E2', 'E2', 'E3', 'E4', 'E4', 'E5'],
    'departmentId': ['D1', 'D1', 'D2', 'D2', 'D3', 'D1', 'D4'],
    'departmentName': ['HR', 'HR', 'Finance', 'Finance', 'IT', 'HR', 'Marketing'],
    'location': ['New York', 'New York', 'London', 'London', 'San Francisco', 'New York', 'Berlin']
}

# Wrap into a dataframe
df = pd.DataFrame(data)

# Run the function 
cross_cluster(df, 'employeeId', 'departmentId', ['departmentName', 'location'])

# Add additional testing 
def test_cross_cluster():
    # Sample data for testing
    data = {
        'employeeId': ['E1', 'E2', 'E3', 'E4'],
        'departmentId': ['D1', 'D2', 'D2', 'D1'],
        'departmentName': ['HR', 'Finance', 'Finance', 'HR'],
        'location': ['New York', 'London', 'London', 'New York']
    }
    df = pd.DataFrame(data)

    # Expected results
    expected_columns = set(['employeeIds', 'departmentIds', 'departmentId_departmentName', 'departmentId_location'])
    
    # Running the function
    result_df = cross_cluster(df, 'employeeId', 'departmentId', ['departmentName', 'location'])
    
    # Asserting the test conditions
    assert set(result_df.columns) == expected_columns, "DataFrame does not have the correct columns"

    # Asserting a specific cluster
    # Note: This assumes that the result is in a specific order, which might not always be true
    expected_cluster = {
        'employeeIds': ['E1', 'E4'],
        'departmentIds': ['D1'],
        'departmentId_departmentName': ['HR'],
        'departmentId_location': ['New York']
    }

    # Convert the first row of DataFrame to dict for easy comparison
    result_cluster = result_df.iloc[0].apply(lambda x: sorted(x) if isinstance(x, list) else x).to_dict()

    assert all(result_cluster[k] == expected_cluster[k] for k in expected_columns), "Cluster data does not match the expected result"

    print("All tests passed.")

# Run the test
test_cross_cluster_function()
