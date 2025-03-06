import numpy as np

def parse_id_components(id_str):
    parts = id_str.split('_')
    
    if id_str == 'total':
        return {'level': 1}
    
    # Level 2: State
    if len(parts) == 1 and parts[0] in ['CA', 'TX', 'WI']:
        return {'level': 2, 'state': parts[0]}
    
    # Level 4: Category
    if len(parts) == 1 and parts[0] in ['FOODS', 'HOBBIES', 'HOUSEHOLD']:
        return {'level': 4, 'category': parts[0]}
    
    # Level 5: Department (예: FOODS_1)
    if len(parts) == 2 and parts[0] in ['FOODS', 'HOBBIES', 'HOUSEHOLD'] and parts[1].isdigit():
        return {'level': 5, 'category': parts[0], 'dept': parts[1]}
    
    # Level 3: Store (예: CA_1)
    if len(parts) == 2 and parts[0] in ['CA', 'TX', 'WI'] and parts[1].isdigit():
        return {'level': 3, 'state': parts[0], 'store': parts[1]}
    
    # Level 6: State-Category (예: CA_FOODS)
    if len(parts) == 2 and parts[0] in ['CA', 'TX', 'WI'] and parts[1] in ['FOODS', 'HOBBIES', 'HOUSEHOLD']:
        return {'level': 6, 'state': parts[0], 'category': parts[1]}
    
    # Level 8: Store-Category (예: CA_1_FOODS)
    if len(parts) == 3 and parts[0] in ['CA', 'TX', 'WI'] and parts[1].isdigit() and parts[2] in ['FOODS', 'HOBBIES', 'HOUSEHOLD']:
        return {'level': 8, 'state': parts[0], 'store': parts[1], 'category': parts[2]}
    
    # Level 10: Product (예: FOODS_1_001)
    if len(parts) == 3 and parts[0] in ['FOODS', 'HOBBIES', 'HOUSEHOLD'] and parts[1].isdigit() and parts[2].isdigit():
        return {'level': 10, 'category': parts[0], 'dept': parts[1], 'item': parts[2]}
    
    # Level 7: State-Department (예: CA_FOODS_1)
    if len(parts) == 3 and parts[0] in ['CA', 'TX', 'WI'] and parts[1] in ['FOODS', 'HOBBIES', 'HOUSEHOLD'] and parts[2].isdigit():
        return {'level': 7, 'state': parts[0], 'category': parts[1], 'dept': parts[2]}
    
    # Level 9: Store-Department (예: CA_1_FOODS_1)
    if len(parts) == 4 and parts[0] in ['CA', 'TX', 'WI'] and parts[1].isdigit() and parts[2] in ['FOODS', 'HOBBIES', 'HOUSEHOLD'] and parts[3].isdigit():
        return {'level': 9, 'state': parts[0], 'store': parts[1], 'category': parts[2], 'dept': parts[3]}
    
    # Level 11: Product-State (예: FOODS_1_001_CA)
    if len(parts) == 4 and parts[0] in ['FOODS', 'HOBBIES', 'HOUSEHOLD'] and parts[1].isdigit() and parts[2].isdigit() and parts[3] in ['CA', 'TX', 'WI']:
        return {'level': 11, 'category': parts[0], 'dept': parts[1], 'item': parts[2], 'state': parts[3]}
    
    # Level 12: Product-Store (예: FOODS_1_001_CA_1)
    if len(parts) == 5 and parts[0] in ['FOODS', 'HOBBIES', 'HOUSEHOLD'] and parts[1].isdigit() and parts[2].isdigit() and parts[3] in ['CA', 'TX', 'WI'] and parts[4].isdigit():
        return {'level': 12, 'category': parts[0], 'dept': parts[1], 'item': parts[2], 'state': parts[3], 'store': parts[4]}
    
    # 매칭되지 않는 경우 에러 출력
    raise ValueError(f"Unknown ID format: {id_str}")


def create_edges(node_mapping):
    edges = []
    edge_types = []
    
    states = ['CA', 'TX', 'WI']
    stores = {
        'CA': ['CA_1', 'CA_2', 'CA_3', 'CA_4'],
        'TX': ['TX_1', 'TX_2', 'TX_3'],
        'WI': ['WI_1', 'WI_2', 'WI_3']
    }
    categories = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
    departments = {
        'FOODS': ['FOODS_1', 'FOODS_2', 'FOODS_3'],
        'HOBBIES': ['HOBBIES_1', 'HOBBIES_2'],
        'HOUSEHOLD': ['HOUSEHOLD_1', 'HOUSEHOLD_2']
    }
    
    ### Geographical Hierarchy
    total_idx = node_mapping['total']
    
    # Total -> State
    for state in states:
        state_idx = node_mapping[state]
        edges.extend([
            [state_idx, total_idx],    # 상향
            [total_idx, state_idx]     # 하향
        ])
        edge_types.extend([0, 1])
        
        # State -> Store
        for store in stores[state]:
            store_idx = node_mapping[store]
            edges.extend([
                [store_idx, state_idx],    # 상향
                [state_idx, store_idx]     # 하향
            ])
            edge_types.extend([0, 1])

    ### Product Hierarchy
    # Total -> Category
    for category in categories:
        cat_idx = node_mapping[category]
        edges.extend([
            [cat_idx, total_idx],    # 상향
            [total_idx, cat_idx]     # 하향
        ])
        edge_types.extend([0, 1])
        
        # Category -> Department
        for dept in departments[category]:
            dept_idx = node_mapping[dept]
            edges.extend([
                [dept_idx, cat_idx],    # 상향
                [cat_idx, dept_idx]     # 하향
            ])
            edge_types.extend([0, 1])
            
            # Department -> Item
            for item in [f"{dept}_{str(i).zfill(3)}" for i in range(1, 400)]:  # 실제 아이템 범위
                if item in node_mapping:  # 존재하는 아이템만 처리
                    item_idx = node_mapping[item]
                    edges.extend([
                        [item_idx, dept_idx],    # 상향
                        [dept_idx, item_idx]     # 하향
                    ])
                    edge_types.extend([0, 1])

    ### Cross
    for state in states:
        state_idx = node_mapping[state]
        
        # State x Category
        for category in categories:
            cat_idx = node_mapping[category]
            cross_id = f"{state}_{category}"
            if cross_id in node_mapping:
                cross_idx = node_mapping[cross_id]
                edges.extend([
                    [cross_idx, state_idx],  # State와 연결
                    [cross_idx, cat_idx]     # Category와 연결
                ])
                edge_types.extend([2, 2])

        # State x Department
        for category in categories:
            for dept in departments[category]:
                dept_idx = node_mapping[dept]
                cross_id = f"{state}_{category}_{dept.split('_')[1]}"
                if cross_id in node_mapping:
                    cross_idx = node_mapping[cross_id]
                    edges.extend([
                        [cross_idx, state_idx],  # State와 연결
                        [cross_idx, dept_idx]    # Department와 연결
                    ])
                    edge_types.extend([2, 2])

        # State x Item
        for category in categories:
            for dept in departments[category]:
                for item in [f"{dept}_{str(i).zfill(3)}" for i in range(1, 400)]:
                    if item in node_mapping:
                        item_idx = node_mapping[item]
                        cross_id = f"{state}_{item}"
                        if cross_id in node_mapping:
                            cross_idx = node_mapping[cross_id]
                            edges.extend([
                                [cross_idx, state_idx],  # State와 연결
                                [cross_idx, item_idx]    # Item과 연결
                            ])
                            edge_types.extend([2, 2])

    for state in states:
        for store in stores[state]:
            store_idx = node_mapping[store]
            
            # Store x Category
            for category in categories:
                cat_idx = node_mapping[category]
                cross_id = f"{store}_{category}"
                if cross_id in node_mapping:
                    cross_idx = node_mapping[cross_id]
                    edges.extend([
                        [cross_idx, store_idx],  # Store와 연결
                        [cross_idx, cat_idx]     # Category와 연결
                    ])
                    edge_types.extend([2, 2])

            # Store x Department
            for category in categories:
                for dept in departments[category]:
                    dept_idx = node_mapping[dept]
                    cross_id = f"{store}_{category}_{dept.split('_')[1]}"
                    if cross_id in node_mapping:
                        cross_idx = node_mapping[cross_id]
                        edges.extend([
                            [cross_idx, store_idx],  # Store와 연결
                            [cross_idx, dept_idx]    # Department와 연결
                        ])
                        edge_types.extend([2, 2])

            # Store x Item
            for category in categories:
                for dept in departments[category]:
                    for item in [f"{dept}_{str(i).zfill(3)}" for i in range(1, 400)]:
                        if item in node_mapping:
                            item_idx = node_mapping[item]
                            cross_id = f"{store}_{item}"
                            if cross_id in node_mapping:
                                cross_idx = node_mapping[cross_id]
                                edges.extend([
                                    [cross_idx, store_idx],  # Store와 연결
                                    [cross_idx, item_idx]    # Item과 연결
                                ])
                                edge_types.extend([2, 2])

    ### Cross Hierarchy
    # State x Category -> State x Department -> State x Item
    for state in states:
        for category in categories:
            state_cat_id = f"{state}_{category}"
            for dept in departments[category]:
                state_dept_id = f"{state}_{category}_{dept.split('_')[1]}"
                if state_cat_id in node_mapping and state_dept_id in node_mapping:
                    edges.extend([
                        [node_mapping[state_dept_id], node_mapping[state_cat_id]],  # 상향
                        [node_mapping[state_cat_id], node_mapping[state_dept_id]]   # 하향
                    ])
                    edge_types.extend([0, 1])
                    
                # State x Department -> State x Item
                for item in [f"{dept}_{str(i).zfill(3)}" for i in range(1, 400)]:
                    if item in node_mapping:
                        state_item_id = f"{state}_{item}"
                        if state_dept_id in node_mapping and state_item_id in node_mapping:
                            edges.extend([
                                [node_mapping[state_item_id], node_mapping[state_dept_id]],  # 상향
                                [node_mapping[state_dept_id], node_mapping[state_item_id]]   # 하향
                            ])
                            edge_types.extend([0, 1])

    # Store x Category -> Store x Department -> Store x Item
    for state in states:
        for store in stores[state]:
            for category in categories:
                store_cat_id = f"{store}_{category}"
                for dept in departments[category]:
                    store_dept_id = f"{store}_{category}_{dept.split('_')[1]}"
                    if store_cat_id in node_mapping and store_dept_id in node_mapping:
                        edges.extend([
                            [node_mapping[store_dept_id], node_mapping[store_cat_id]],  # 상향
                            [node_mapping[store_cat_id], node_mapping[store_dept_id]]   # 하향
                        ])
                        edge_types.extend([0, 1])
                        
                    # Store x Department -> Store x Item
                    for item in [f"{dept}_{str(i).zfill(3)}" for i in range(1, 400)]:
                        if item in node_mapping:
                            store_item_id = f"{store}_{item}"
                            if store_dept_id in node_mapping and store_item_id in node_mapping:
                                edges.extend([
                                    [node_mapping[store_item_id], node_mapping[store_dept_id]],  # 상향
                                    [node_mapping[store_dept_id], node_mapping[store_item_id]]   # 하향
                                ])
                                edge_types.extend([0, 1])

    # State x Category -> Store x Category
    # State x Department -> Store x Department
    # State x Item -> Store x Item
    for state in states:
        for store in stores[state]:
            # State x Category -> Store x Category
            for category in categories:
                state_cat_id = f"{state}_{category}"
                store_cat_id = f"{store}_{category}"
                if state_cat_id in node_mapping and store_cat_id in node_mapping:
                    edges.extend([
                        [node_mapping[store_cat_id], node_mapping[state_cat_id]],  # 상향
                        [node_mapping[state_cat_id], node_mapping[store_cat_id]]   # 하향
                    ])
                    edge_types.extend([0, 1])

            # State x Department -> Store x Department
            for category in categories:
                for dept in departments[category]:
                    state_dept_id = f"{state}_{category}_{dept.split('_')[1]}"
                    store_dept_id = f"{store}_{category}_{dept.split('_')[1]}"
                    if state_dept_id in node_mapping and store_dept_id in node_mapping:
                        edges.extend([
                            [node_mapping[store_dept_id], node_mapping[state_dept_id]],  # 상향
                            [node_mapping[state_dept_id], node_mapping[store_dept_id]]   # 하향
                        ])
                        edge_types.extend([0, 1])

            # State x Item -> Store x Item
            for category in categories:
                for dept in departments[category]:
                    for item in [f"{dept}_{str(i).zfill(3)}" for i in range(1, 400)]:
                        if item in node_mapping:
                            state_item_id = f"{state}_{item}"
                            store_item_id = f"{store}_{item}"
                            if state_item_id in node_mapping and store_item_id in node_mapping:
                                edges.extend([
                                    [node_mapping[store_item_id], node_mapping[state_item_id]],  # 상향
                                    [node_mapping[state_item_id], node_mapping[store_item_id]]   # 하향
                                ])
                                edge_types.extend([0, 1])

    return np.array(edges), np.array(edge_types)   