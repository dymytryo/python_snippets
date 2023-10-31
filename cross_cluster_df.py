def cross_cluster(dataframe: pd.DataFrame, column_a: str, column_b: str) -> pd.DataFrame:
    """
    Create clusters based on the connections between two types of entities in a given DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the relationship data.
        column_a (str): The name of the first column in the DataFrame representing the first entity.
        column_b (str): The name of the second column in the DataFrame representing the second entity.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a cluster with two lists of entity IDs from each column.

    Raises:
        ValueError: If the specified columns are not found in the DataFrame.

    Example:
        >>> df = pd.DataFrame({'employeeId': ['E1', 'E2', 'E2'], 'departmentId': ['D5', 'D5', 'D6']})
        >>> create_clusters(df, 'employeeId', 'departmentId')
    """
    if column_a not in dataframe.columns or column_b not in dataframe.columns:
        raise ValueError("Specified columns not found in the DataFrame")

    G = nx.Graph()
    for _, row in dataframe.iterrows():
        entity_a_id = str(row[column_a])
        entity_b_id = str(row[column_b])

        entity_a_node = f'a{entity_a_id}'
        entity_b_node = f'b{entity_b_id}'
        G.add_edge(entity_a_node, entity_b_node)

    connected_components = list(nx.connected_components(G))
    clusters = []

    for component in connected_components:
        entities_a = {n[1:] for n in component if n.startswith('a')}
        entities_b = {n[1:] for n in component if n.startswith('b')}
        clusters.append({f'{column_a}s': list(entities_a), f'{column_b}s': list(entities_b)})

    return pd.DataFrame(clusters)
