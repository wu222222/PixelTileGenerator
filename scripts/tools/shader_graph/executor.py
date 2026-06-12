"""
Graph Executor — 拓扑排序 + 执行引擎
"""

from .nodes import create_node, NODE_REGISTRY


class GraphExecutor:
    """
    执行节点图

    输入: JSON graph (nodes + edges)
    输出: 执行结果 (output 节点的 image)
    """

    def __init__(self, model_context=None):
        """
        Args:
            model_context: dict 包含 _model, _device, _latents, _cluster_reps 等
        """
        self.context = model_context or {}

    def execute(self, graph_json):
        """
        执行图

        Args:
            graph_json: {
                "nodes": [{"id": ..., "type": ..., "params": ...}, ...],
                "edges": [{"from": ..., "to": ..., "slot": ...}, ...]
            }

        Returns:
            dict: {output_node_id: PIL.Image}
        """
        nodes_def = graph_json.get("nodes", [])
        edges = graph_json.get("edges", [])

        # 创建节点实例
        nodes = {}
        for nd in nodes_def:
            node = create_node(nd["type"], nd["id"], nd.get("params", {}))
            nodes[nd.id if hasattr(nd, 'id') else nd["id"]] = node

        # 建立连接映射: to_node_id → {slot_name: from_node_id}
        conn_map = {}
        for edge in edges:
            to_id = edge["to"]
            from_id = edge["from"]
            slot = edge.get("slot", "default")
            if to_id not in conn_map:
                conn_map[to_id] = {}
            conn_map[to_id][slot] = from_id

        # 拓扑排序
        sorted_nodes = self._topological_sort(nodes_def, edges)

        # 执行
        context = dict(self.context)
        for node_id in sorted_nodes:
            if node_id not in nodes:
                continue
            node = nodes[node_id]
            node.inputs_connected = conn_map.get(node_id, {})
            node.execute(context)

        # 收集输出
        outputs = {}
        for node_id, node in nodes.items():
            if node.NODE_TYPE == "output" and node_id in context:
                outputs[node_id] = context[node_id].get("image")

        return outputs

    def _topological_sort(self, nodes_def, edges):
        """拓扑排序"""
        node_ids = [n["id"] if isinstance(n, dict) else n.id for n in nodes_def]
        in_degree = {nid: 0 for nid in node_ids}
        adj = {nid: [] for nid in node_ids}

        for edge in edges:
            adj[edge["from"]].append(edge["to"])
            in_degree[edge["to"]] = in_degree.get(edge["to"], 0) + 1

        queue = [nid for nid in node_ids if in_degree[nid] == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            for neighbor in adj.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result
