from graphviz import Digraph

def add_node(dot, tree, parent=None):
    if 'split_feature' in tree:
        split_feature = tree['split_feature']
        split_value = tree['split_value']
        node_id = f"{split_feature}_{split_value}"
        dot.node(node_id, f"{split_feature} <= {split_value}")
        if parent:
            dot.edge(parent, node_id)
        add_node(dot, tree['left_tree'], node_id)
        add_node(dot, tree['right_tree'], node_id)
    elif 'leaf_value' in tree:
        leaf_value = tree['leaf_value']
        node_id = f"leaf_{leaf_value}"
        dot.node(node_id, f"Leaf: {leaf_value}", style='filled', color='lightblue')
        if parent:
            dot.edge(parent, node_id)

def visualize_tree(tree_data):
    dot = Digraph(comment='Decision Tree')
    add_node(dot, tree_data)
    return dot

# 示例数据
tree_data = {
    "split_feature": "label",
    "split_value": 28.90303030303031,
    "left_tree": {
        "split_feature": "label",
        "split_value": 20.8,
        "left_tree": {
            "split_feature": "label",
            "split_value": 16.6,
            "left_tree": {
                "leaf_value": 14.654838709677419
            },
            "right_tree": {
                "split_feature": "label",
                "split_value": 18.9,
                "left_tree": {
                    "leaf_value": 18.181481481481484
                },
                "right_tree": {
                    "leaf_value": 19.962857142857143
                }
            }
        },
        "right_tree": {
            "split_feature": "label",
            "split_value": 25.0,
            "left_tree": {
                "split_feature": "label",
                "split_value": 22.9,
                "left_tree": {
                    "leaf_value": 21.975000000000005
                },
                "right_tree": {
                    "leaf_value": 23.98
                }
            },
            "right_tree": {
                "leaf_value": 27.37857142857143
            }
        }
    },
    "right_tree": {
        "split_feature": "label",
        "split_value": 39.8,
        "left_tree": {
            "split_feature": "label",
            "split_value": 33.8,
            "left_tree": {
                "leaf_value": 31.39375
            },
            "right_tree": {
                "leaf_value": 36.39565217391304
            }
        },
        "right_tree": {
            "leaf_value": 47.089999999999996
        }
    }
}

dot = visualize_tree(tree_data)
dot.render('decision_tree', format='png', view=True)