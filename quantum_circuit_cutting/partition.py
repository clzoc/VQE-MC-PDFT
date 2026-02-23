"""Qubit partitioning for circuit cutting.

Partitions n logical qubits into clusters of at most d qubits each,
minimizing inter-cluster entangling gates. Three strategies:

1. Symmetry-guided: groups qubits by MO symmetry label (best for
   chemical systems with known orbital symmetry).
2. Spectral bisection: uses Fiedler vector of the graph Laplacian
   for balanced partitioning, recursively applied.
3. Greedy + Kernighan-Lin refinement: BFS seed + KL local search
   to reduce cut edges.

Reference: Peng et al., PRL 125, 150504 (2020).
Inspired by: qiskit-addon-cutting partition strategies.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterPartition:
    """Result of circuit partitioning.

    Attributes:
        clusters: List of qubit index sets, one per cluster.
        inter_cluster_edges: List of (qubit_i, qubit_j) pairs cut.
        cluster_graph: NetworkX graph where nodes=clusters, edges=cuts.
        n_cuts: Total number of inter-cluster wire cuts K.
    """

    clusters: List[List[int]]
    inter_cluster_edges: List[Tuple[int, int]]
    cluster_graph: nx.Graph
    n_cuts: int

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def max_cluster_size(self) -> int:
        return max(len(c) for c in self.clusters) if self.clusters else 0

    def qubit_to_cluster(self) -> Dict[int, int]:
        """Map each qubit to its cluster index."""
        mapping: Dict[int, int] = {}
        for cluster_idx, qubits in enumerate(self.clusters):
            for q in qubits:
                mapping[q] = cluster_idx
        return mapping

    @property
    def overhead_kappa(self) -> float:
        """Sampling overhead kappa = 4^K for wire cuts."""
        return 4.0**self.n_cuts


def _build_interaction_graph(
    n_qubits: int, entangling_gates: List[Tuple[int, int]]
) -> nx.Graph:
    """Build weighted undirected qubit interaction graph from gate list."""
    g = nx.Graph()
    g.add_nodes_from(range(n_qubits))
    for q_i, q_j in entangling_gates:
        if g.has_edge(q_i, q_j):
            g[q_i][q_j]["weight"] += 1
        else:
            g.add_edge(q_i, q_j, weight=1)
    return g


def _build_cluster_partition(
    clusters: List[List[int]], 
    entangling_gates: List[Tuple[int, int]],
    interaction_graph: nx.Graph
) -> ClusterPartition:
    """Build ClusterPartition from cluster assignment and original entangling gates.
    Preserves directed edge order from original entangling gates (control -> target).
    """
    q2c: Dict[int, int] = {}
    for ci, qubits in enumerate(clusters):
        for q in qubits:
            q2c[q] = ci

    inter_edges: List[Tuple[int, int]] = []
    seen_edges = set()
    for q_control, q_target in entangling_gates:
        # Only count each unique inter-cluster edge once, preserve direction
        if q2c.get(q_control, -1) != q2c.get(q_target, -2):
            edge = (q_control, q_target)
            if edge not in seen_edges:
                inter_edges.append(edge)
                seen_edges.add(edge)
                seen_edges.add((q_target, q_control))  # Avoid reverse duplicates

    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from(range(len(clusters)))
    for q_i, q_j in inter_edges:
        ci, cj = q2c[q_i], q2c[q_j]
        if cluster_graph.has_edge(ci, cj):
            cluster_graph[ci][cj]["weight"] += 1
        else:
            cluster_graph.add_edge(ci, cj, weight=1)

    return ClusterPartition(
        clusters=clusters,
        inter_cluster_edges=inter_edges,
        cluster_graph=cluster_graph,
        n_cuts=len(inter_edges),
    )


class CircuitPartitioner:
    """Partitions quantum circuits into clusters for cutting.

    Builds a qubit interaction graph from entangling gates and partitions
    it into clusters of bounded size, minimizing the number of cuts K.

    Strategies (in order of preference):
    1. Symmetry-guided (if orbital_symmetries provided)
    2. Spectral bisection (default for large circuits)
    3. Greedy BFS + KL refinement (fallback)

    All strategies are followed by Kernighan-Lin refinement to locally
    minimize the cut count.

    Args:
        n_qubits: Total number of logical qubits.
        max_cluster_size: Maximum qubits per cluster (device limit d).
    """

    def __init__(self, n_qubits: int, max_cluster_size: int = 13,
                 strategy: str = "spectral"):
        """
        Args:
            n_qubits: Total number of logical qubits.
            max_cluster_size: Maximum qubits per cluster (device limit d).
            strategy: Partitioning strategy. One of:
                - ``"spectral"`` (default): Spectral bisection + KL refinement.
                - ``"best_first"``: Best-first search with pruning.
                  Explores partial assignments ordered by a lower-bound
                  on the cut count, pruning branches that cannot improve
                  on the best solution found so far.  Inspired by the
                  automated cut finding in Qiskit Circuit Knitting Toolbox.
        """
        self.n_qubits = n_qubits
        self.max_cluster_size = max_cluster_size
        self.strategy = strategy

    def partition(
        self,
        entangling_gates: List[Tuple[int, int]],
        orbital_symmetries: Dict[int, str] | None = None,
    ) -> ClusterPartition:
        """Partition qubits into clusters minimizing inter-cluster cuts.

        Args:
            entangling_gates: List of (qubit_i, qubit_j) for each
                two-qubit gate in the circuit.
            orbital_symmetries: Optional dict mapping qubit index to
                MO symmetry label for symmetry-guided partitioning.

        Returns:
            ClusterPartition with clusters, cuts, and cluster graph.
        """
        if self.n_qubits <= self.max_cluster_size:
            # No cutting needed
            return _build_cluster_partition(
                [list(range(self.n_qubits))],
                entangling_gates,
                _build_interaction_graph(self.n_qubits, entangling_gates),
            )

        graph = _build_interaction_graph(self.n_qubits, entangling_gates)

        if orbital_symmetries is not None:
            clusters = self._symmetry_partition(graph, orbital_symmetries)
        elif self.strategy == "best_first":
            clusters = self._best_first_partition(graph, orbital_symmetries)
        else:
            clusters = self._spectral_partition(graph)

        # Kernighan-Lin refinement
        clusters = _kl_refine(clusters, graph, self.max_cluster_size)

        result = _build_cluster_partition(clusters, entangling_gates, graph)
        logger.info(
            "Partitioned %d qubits -> %d clusters (max %d), %d cuts, kappa=%.1f",
            self.n_qubits,
            result.n_clusters,
            result.max_cluster_size,
            result.n_cuts,
            result.overhead_kappa,
        )
        return result

    def _spectral_partition(self, graph: nx.Graph) -> List[List[int]]:
        """Recursive spectral bisection using the Fiedler vector.

        The Fiedler vector (2nd smallest eigenvector of the graph
        Laplacian) provides an optimal 2-way partition. We recursively
        bisect until all clusters are within the size limit.
        """
        all_nodes = sorted(graph.nodes())
        return self._recursive_bisect(graph, all_nodes)

    def _recursive_bisect(self, graph: nx.Graph, nodes: List[int]) -> List[List[int]]:
        """Recursively bisect a node set using spectral methods."""
        if len(nodes) <= self.max_cluster_size:
            return [sorted(nodes)]

        subgraph = graph.subgraph(nodes)

        # Use Fiedler vector for bisection if graph is connected
        if nx.is_connected(subgraph) and len(nodes) > 2:
            try:
                fiedler = nx.fiedler_vector(subgraph, weight="weight")
                node_list = sorted(subgraph.nodes())
                # Split by sign of Fiedler vector
                part_a = [node_list[i] for i, v in enumerate(fiedler) if v <= 0]
                part_b = [node_list[i] for i, v in enumerate(fiedler) if v > 0]
            except (nx.NetworkXError, np.linalg.LinAlgError):
                mid = len(nodes) // 2
                part_a, part_b = nodes[:mid], nodes[mid:]
        else:
            # Disconnected or too small: split by connected components or midpoint
            components = list(nx.connected_components(subgraph))
            if len(components) >= 2:
                part_a = sorted(components[0])
                part_b = sorted(set(nodes) - components[0])
            else:
                mid = len(nodes) // 2
                part_a, part_b = nodes[:mid], nodes[mid:]

        # Ensure neither partition is empty
        if not part_a or not part_b:
            mid = len(nodes) // 2
            part_a, part_b = nodes[:mid], nodes[mid:]

        return self._recursive_bisect(graph, part_a) + self._recursive_bisect(
            graph, part_b
        )

    def _symmetry_partition(
        self,
        graph: nx.Graph,
        symmetries: Dict[int, str],
    ) -> List[List[int]]:
        """Partition guided by MO symmetry labels.

        Groups qubits with the same symmetry label together, then
        splits oversized groups to respect the device limit.
        """
        sym_groups: Dict[str, List[int]] = {}
        for q, sym in symmetries.items():
            sym_groups.setdefault(sym, []).append(q)

        for q in range(self.n_qubits):
            if q not in symmetries:
                sym_groups.setdefault("ungrouped", []).append(q)

        clusters: List[List[int]] = []
        for qubits in sym_groups.values():
            qubits_sorted = sorted(qubits)
            for i in range(0, len(qubits_sorted), self.max_cluster_size):
                clusters.append(qubits_sorted[i : i + self.max_cluster_size])

        return clusters

    def _greedy_partition(self, graph: nx.Graph) -> List[List[int]]:
        """Greedy partitioning minimizing inter-cluster edges.

        Uses a BFS-based approach: start from highest-degree node,
        grow cluster until size limit, then start new cluster.
        """
        assigned: Set[int] = set()
        clusters: List[List[int]] = []

        nodes_by_degree = sorted(
            graph.nodes(), key=lambda n: graph.degree(n), reverse=True
        )

        for start_node in nodes_by_degree:
            if start_node in assigned:
                continue

            cluster: List[int] = [start_node]
            assigned.add(start_node)

            # BFS expansion prioritizing high-connectivity neighbors
            candidates = list(graph.neighbors(start_node))
            candidates.sort(
                key=lambda n: sum(1 for nb in graph.neighbors(n) if nb in set(cluster)),
                reverse=True,
            )

            for candidate in candidates:
                if len(cluster) >= self.max_cluster_size:
                    break
                if candidate not in assigned:
                    cluster.append(candidate)
                    assigned.add(candidate)

            clusters.append(sorted(cluster))

        for q in range(self.n_qubits):
            if q not in assigned:
                if clusters and len(clusters[-1]) < self.max_cluster_size:
                    clusters[-1].append(q)
                else:
                    clusters.append([q])

        return clusters

    def _best_first_partition(
        self,
        graph: nx.Graph,
        orbital_symmetries: Optional[Dict[int, str]] = None,
    ) -> List[List[int]]:
        """Best-first search cut finding with pruning.

        Explores partial qubit-to-cluster assignments using a priority
        queue ordered by a lower bound on the final cut count.  Prunes
        any branch whose lower bound exceeds the best complete solution
        found so far.

        When *orbital_symmetries* is provided, the heuristic biases
        assignment toward placing qubits with the same symmetry label
        into the same cluster, reducing the effective search space.

        Reference: Tang et al., "CutQC" (2021); Qiskit Circuit Knitting
        Toolbox ``automated_cut_finding``.
        """
        nodes = sorted(graph.nodes())
        n = len(nodes)
        n_clusters_needed = math.ceil(n / self.max_cluster_size)

        # Seed the search with a spectral solution as the initial upper bound
        seed_clusters = self._spectral_partition(graph)
        seed_clusters = _kl_refine(seed_clusters, graph, self.max_cluster_size)
        best_cuts = _cut_count(seed_clusters, graph)
        best_clusters = seed_clusters

        # Build symmetry affinity: prefer grouping same-symmetry qubits
        sym_label: Dict[int, str] = orbital_symmetries or {}

        # --- priority queue entries: (lower_bound, tie_break, assignment) ---
        # assignment: dict  qubit -> cluster_idx  (partial)
        initial: Dict[int, int] = {}
        lb0 = self._cut_lower_bound(initial, graph, n_clusters_needed)
        counter = 0  # tie-breaker for heap stability
        heap: list = [(lb0, counter, initial)]

        max_expansions = 50_000  # safety cap

        while heap and max_expansions > 0:
            max_expansions -= 1
            lb, _, assignment = heapq.heappop(heap)

            if lb >= best_cuts:
                continue  # prune

            # Pick next unassigned qubit (highest-degree first for tighter bounds)
            unassigned = [q for q in nodes if q not in assignment]
            if not unassigned:
                # Complete assignment – evaluate
                clusters = self._assignment_to_clusters(assignment, n_clusters_needed)
                cuts = _cut_count(clusters, graph)
                if cuts < best_cuts:
                    best_cuts = cuts
                    best_clusters = clusters
                continue

            # Choose qubit with most already-assigned neighbours (most constrained)
            next_q = max(
                unassigned,
                key=lambda q: sum(1 for nb in graph.neighbors(q) if nb in assignment),
            )

            # Determine candidate cluster ids
            candidate_cids: List[int] = []
            # Existing clusters that still have room
            cluster_sizes: Dict[int, int] = {}
            for cid in assignment.values():
                cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
            for cid in range(n_clusters_needed):
                if cluster_sizes.get(cid, 0) < self.max_cluster_size:
                    candidate_cids.append(cid)

            # Sort candidates: prefer cluster containing same-symmetry qubits
            q_sym = sym_label.get(next_q)
            def _affinity(cid: int) -> int:
                if q_sym is None:
                    return 0
                return -sum(
                    1 for q2, c2 in assignment.items()
                    if c2 == cid and sym_label.get(q2) == q_sym
                )
            candidate_cids.sort(key=_affinity)

            for cid in candidate_cids:
                child = dict(assignment)
                child[next_q] = cid
                child_lb = self._cut_lower_bound(child, graph, n_clusters_needed)
                if child_lb < best_cuts:
                    counter += 1
                    heapq.heappush(heap, (child_lb, counter, child))

        logger.info(
            "Best-first search: %d cuts (seed had %d)",
            best_cuts,
            _cut_count(seed_clusters, graph),
        )
        return best_clusters

    # ------------------------------------------------------------------
    # helpers for best-first search
    # ------------------------------------------------------------------

    def _cut_lower_bound(
        self,
        assignment: Dict[int, int],
        graph: nx.Graph,
        n_clusters: int,
    ) -> int:
        """Admissible lower bound on the final cut count.

        Counts edges that are *already* cut (both endpoints assigned to
        different clusters) plus, for each edge with exactly one assigned
        endpoint, a fractional contribution reflecting the probability
        that the other endpoint lands in a different cluster.
        """
        already_cut = 0
        maybe_cut = 0
        for u, v in graph.edges():
            cu = assignment.get(u)
            cv = assignment.get(v)
            if cu is not None and cv is not None:
                if cu != cv:
                    already_cut += 1
            elif cu is not None or cv is not None:
                # One assigned, one not – at least (1 - 1/n_clusters) chance of cut
                maybe_cut += 1
        # Admissible: count definite cuts + floor of expected uncertain cuts
        return already_cut + (maybe_cut * (n_clusters - 1)) // n_clusters

    @staticmethod
    def _assignment_to_clusters(
        assignment: Dict[int, int], n_clusters: int
    ) -> List[List[int]]:
        clusters: List[List[int]] = [[] for _ in range(n_clusters)]
        for q, cid in assignment.items():
            clusters[cid].append(q)
        return [sorted(c) for c in clusters if c]

    @staticmethod
    def cr2_symmetry_labels(n_spatial_orbitals: int) -> Dict[int, str]:
        """Generate MO symmetry labels for Cr2 active space.

        For Cr2, the active orbitals have D_inf_h symmetry:
        sigma_g, sigma_u, pi_gx, pi_gy, pi_ux, pi_uy, delta_gx, delta_gy, ...

        Returns:
            Dict mapping spin-orbital qubit index to symmetry label.
        """
        sym_cycle = [
            "sigma_g",
            "sigma_u",
            "pi_ux",
            "pi_uy",
            "pi_gx",
            "pi_gy",
            "delta_gx",
            "delta_gy",
            "delta_ux",
            "delta_uy",
            "sigma_g2",
            "sigma_u2",
        ]
        labels: Dict[int, str] = {}
        for i in range(2 * n_spatial_orbitals):
            spatial_idx = i // 2
            sym_idx = spatial_idx % len(sym_cycle)
            labels[i] = sym_cycle[sym_idx]
        return labels


def _kl_refine(
    clusters: List[List[int]],
    graph: nx.Graph,
    max_size: int,
    max_passes: int = 10,
) -> List[List[int]]:
    """Kernighan-Lin style refinement to reduce inter-cluster cuts.

    For each pair of adjacent clusters, tries swapping boundary nodes
    to reduce the total cut count. Repeats until no improvement.

    This is a simplified KL that respects the max_size constraint.
    """
    clusters = [list(c) for c in clusters]  # deep copy

    for _ in range(max_passes):
        improved = False
        for ci in range(len(clusters)):
            for cj in range(ci + 1, len(clusters)):
                if _try_swap(clusters, ci, cj, graph, max_size):
                    improved = True
        if not improved:
            break

    # Remove empty clusters
    return [sorted(c) for c in clusters if c]


def _cut_count(clusters: List[List[int]], graph: nx.Graph) -> int:
    """Count inter-cluster edges."""
    q2c: Dict[int, int] = {}
    for ci, qubits in enumerate(clusters):
        for q in qubits:
            q2c[q] = ci
    count = 0
    for u, v in graph.edges():
        if q2c.get(u, -1) != q2c.get(v, -2):
            count += 1
    return count


def _try_swap(
    clusters: List[List[int]],
    ci: int,
    cj: int,
    graph: nx.Graph,
    max_size: int,
) -> bool:
    """Try swapping one boundary node between clusters ci and cj.

    Returns True if an improving swap was made.
    """
    baseline = _cut_count(clusters, graph)

    # Find boundary nodes: nodes in ci adjacent to nodes in cj (and vice versa)
    set_j = set(clusters[cj])
    set_i = set(clusters[ci])
    boundary_i = [
        q for q in clusters[ci] if any(nb in set_j for nb in graph.neighbors(q))
    ]
    boundary_j = [
        q for q in clusters[cj] if any(nb in set_i for nb in graph.neighbors(q))
    ]

    best_gain = 0
    best_move = None  # (node, from_cluster, to_cluster)

    # Try moving a node from ci to cj
    for q in boundary_i:
        if len(clusters[cj]) < max_size and len(clusters[ci]) > 1:
            clusters[ci].remove(q)
            clusters[cj].append(q)
            new_cuts = _cut_count(clusters, graph)
            gain = baseline - new_cuts
            if gain > best_gain:
                best_gain = gain
                best_move = (q, cj, ci)  # record reverse to undo
            clusters[cj].remove(q)
            clusters[ci].append(q)

    # Try moving a node from cj to ci
    for q in boundary_j:
        if len(clusters[ci]) < max_size and len(clusters[cj]) > 1:
            clusters[cj].remove(q)
            clusters[ci].append(q)
            new_cuts = _cut_count(clusters, graph)
            gain = baseline - new_cuts
            if gain > best_gain:
                best_gain = gain
                best_move = (q, ci, cj)
            clusters[ci].remove(q)
            clusters[cj].append(q)

    if best_move is not None:
        q, to_c, from_c = best_move
        clusters[from_c].remove(q)
        clusters[to_c].append(q)
        return True
    return False
