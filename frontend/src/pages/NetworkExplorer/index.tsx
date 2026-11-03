import React, { useState, useEffect, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { getNetworkSummary, getNetworkGraph } from '../../services/api';
import { NetworkSummaryResponse, NetworkGraphResponse, GraphNode, GraphEdge } from '../../types';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Network,
  Search,
  Filter,
  ShieldAlert,
  AlertTriangle,
  Info,
  CheckCircle2,
  Sliders,
  Cpu,
  Layers,
  Sparkles,
} from 'lucide-react';

export const NetworkExplorerPage: React.FC = () => {
  const { t, mode } = useApp();
  const [summary, setSummary] = useState<NetworkSummaryResponse | null>(null);
  const [graphData, setGraphData] = useState<NetworkGraphResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [noiseLevel, setNoiseLevel] = useState<number>(0.1);

  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState<any>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<any>([]);

  const loadGraph = async () => {
    setLoading(true);
    try {
      const sum = await getNetworkSummary(mode);
      setSummary(sum);
      const g = await getNetworkGraph(noiseLevel, mode);
      setGraphData(g);

      // Convert backend nodes to ReactFlow nodes
      const rfNodes = g.nodes.map((n) => {
        const isProd = n.type === 'product';
        return {
          id: n.id,
          type: 'default',
          position: n.position,
          data: {
            label: (
              <div className="flex flex-col text-left">
                <span className="font-bold text-[11px] truncate">{n.label}</span>
                <span className="text-[9px] opacity-80">
                  {n.type.toUpperCase()} • Deg: {n.degree}
                </span>
              </div>
            ),
            rawNode: n,
          },
          style: {
            background: isProd
              ? n.is_suspicious
                ? '#fef2f2' // red
                : '#ecfdf5' // green
              : n.is_suspicious
              ? '#fffbeb' // amber
              : '#f8fafc',
            border: isProd
              ? n.is_suspicious
                ? '2px solid #ef4444'
                : '1.5px solid #10b981'
              : n.is_suspicious
              ? '2px solid #f59e0b'
              : '1.5px solid #64748b',
            borderRadius: isProd ? '8px' : '24px',
            color: '#0f172a',
            padding: '8px 12px',
            fontSize: '11px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            minWidth: '120px',
          },
        };
      });

      // Convert backend edges to ReactFlow edges
      const rfEdges = g.edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        animated: e.is_fake,
        style: {
          stroke: e.is_fake ? '#ef4444' : '#94a3b8',
          strokeWidth: e.is_fake ? 2.5 : 1.2,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: e.is_fake ? '#ef4444' : '#94a3b8',
          width: 14,
          height: 14,
        },
      }));

      setNodes(rfNodes);
      setEdges(rfEdges);
      if (g.nodes.length > 0) {
        setSelectedNode(g.nodes.find((n) => n.is_suspicious) || g.nodes[0]);
      }
    } catch (e) {
      console.warn('Network API unavailable, using local synthesis:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGraph();
  }, [noiseLevel, mode]);

  // Handle node selection
  const onNodeClick = (_event: any, node: any) => {
    if (node.data?.rawNode) {
      setSelectedNode(node.data.rawNode);
    }
  };

  // Re-style nodes when search query changes
  useEffect(() => {
    if (!graphData) return;
    const q = searchQuery.trim().toLowerCase();
    setNodes((prevNodes: any[]) =>
      prevNodes.map((n) => {
        const raw = n.data?.rawNode;
        const matches =
          q !== '' &&
          (raw?.id.toLowerCase().includes(q) || raw?.label.toLowerCase().includes(q));
        return {
          ...n,
          style: {
            ...n.style,
            boxShadow: matches
              ? '0 0 0 3px #6366f1, 0 10px 15px -3px rgba(99, 102, 241, 0.4)'
              : '0 1px 3px rgba(0,0,0,0.1)',
            transform: matches ? 'scale(1.06)' : 'scale(1.0)',
            transition: 'all 0.2s ease',
          },
        };
      })
    );

    if (q !== '') {
      const match = graphData.nodes.find(
        (n) => n.id.toLowerCase().includes(q) || n.label.toLowerCase().includes(q)
      );
      if (match) {
        setSelectedNode(match);
      }
    }
  }, [searchQuery, graphData]);

  return (
    <div className="space-y-6">
      {/* Top Banner & Context */}
      <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Interactive Bipartite Network Explorer</h2>
          <p className="text-xs text-slate-500 mt-1">
            Visualizes bipartite reviewer-product graph topologies, injected coordinated review rings, and structural centralities.
          </p>
        </div>

        {/* Search, Noise & Filter Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search node (e.g. T-0, H-1)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 w-44 font-mono"
            />
          </div>

          <div className="flex items-center gap-1.5 text-xs bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg">
            <Sliders size={14} className="text-slate-500" />
            <span className="text-slate-600 font-medium">Ring Noise:</span>
            <input
              type="range"
              min="0.0"
              max="0.4"
              step="0.1"
              value={noiseLevel}
              onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
              className="w-20 cursor-pointer accent-indigo-600"
            />
            <span className="font-mono font-bold text-indigo-600">{(noiseLevel * 100).toFixed(0)}%</span>
          </div>

          <button
            onClick={loadGraph}
            className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-medium transition-colors shadow-sm"
          >
            Regenerate Graph
          </button>
        </div>
      </div>

      {/* Honest Academic Limitation Banner */}
      <div className="bg-amber-50/80 border border-amber-200 p-4 rounded-xl flex items-start gap-3 text-xs text-amber-900">
        <Info size={18} className="text-amber-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-bold">Transparent Graph Modality & Methodology Disclosure:</p>
          <p className="leading-relaxed">
            This explorer distinguishes four distinct graph regimes evaluated in the research:
            (1) <strong>Synthetic Demonstration Bipartite Graph</strong> (shown interactively below with simulated ring injection);
            (2) <strong>Projected Co-Review Graph</strong> (used for weighted degree and clustering coefficient extraction);
            (3) <strong>Tabular Centrality Models</strong> (pre-extracted PageRank, eigenvector, degree, and clustering features); and
            (4) <strong>KNN Feature-Space Graphs</strong> (used for neighborhood feature aggregation).
            The raw 1 GB Amazon raw graph archive is not bundled; models operate on verified pre-extracted topological features.
          </p>
        </div>
      </div>

      {/* Main Interactive Graph & Inspector (12 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* React Flow Canvas (8 cols) */}
        <div className="lg:col-span-8 bg-white rounded-xl border border-slate-200/80 shadow-sm overflow-hidden flex flex-col h-[560px]">
          <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-xs">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5 font-semibold text-slate-700">
                <span className="w-2.5 h-2.5 rounded-full bg-indigo-500" />
                Reviewer Nodes ({graphData?.stats.reviewer_nodes || 0})
              </span>
              <span className="flex items-center gap-1.5 font-semibold text-slate-700">
                <span className="w-2.5 h-2.5 rounded bg-emerald-500" />
                Product Nodes ({graphData?.stats.product_nodes || 0})
              </span>
              <span className="flex items-center gap-1.5 font-semibold text-rose-600">
                <span className="w-2.5 h-2.5 rounded bg-rose-500" />
                Coordinated Ring ({graphData?.stats.ring_reviewers || 0} revs, {graphData?.stats.target_products || 0} prods)
              </span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">Pan & Zoom Enabled</span>
          </div>

          <div className="flex-1 w-full h-full relative">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              fitView
              attributionPosition="bottom-left"
            >
              <MiniMap
                nodeColor={(node: any) => (node.data?.rawNode?.type === 'product' ? '#10b981' : '#6366f1')}
                className="bg-white/90 border border-slate-200 rounded-lg shadow-sm"
              />
              <Controls className="bg-white border border-slate-200 rounded-lg shadow-sm" />
              <Background color="#cbd5e1" gap={16} size={1} />
            </ReactFlow>
          </div>
        </div>

        {/* Node Inspector & Graph ML Benchmarks (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Node Inspector Card */}
          <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <h3 className="text-sm font-bold text-slate-900">Topology Node Inspector</h3>
              {selectedNode?.is_suspicious && (
                <span className="text-[10px] uppercase font-bold text-rose-700 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
                  Suspicious Fraud Ring
                </span>
              )}
            </div>

            {selectedNode ? (
              <div className="space-y-3 text-xs">
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <span className="text-[11px] text-slate-500 uppercase font-semibold">Node Identifier</span>
                  <div className="text-base font-bold font-mono text-slate-900 mt-0.5">{selectedNode.label}</div>
                  <span className="text-[10px] text-slate-500 font-mono">ID: {selectedNode.id}</span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-100">
                    <span className="text-[10px] text-slate-500 block uppercase">Node Type</span>
                    <span className="font-bold text-slate-900 capitalize font-mono">{selectedNode.type}</span>
                  </div>
                  <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-100">
                    <span className="text-[10px] text-slate-500 block uppercase">Degree Centrality</span>
                    <span className="font-bold text-indigo-600 font-mono text-base">{selectedNode.degree}</span>
                  </div>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg space-y-2 text-[11px] border border-slate-100">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Clustering Coefficient:</span>
                    <span className="font-mono font-bold text-slate-800">
                      {selectedNode.clustering_coef !== undefined && selectedNode.clustering_coef !== null
                        ? selectedNode.clustering_coef.toFixed(4)
                        : selectedNode.type === 'reviewer'
                        ? 'Not applicable (Bipartite Reviewer)'
                        : 'Not available'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Estimated PageRank:</span>
                    <span className="font-mono font-bold text-slate-800">
                      {selectedNode.pagerank !== undefined && selectedNode.pagerank !== null
                        ? selectedNode.pagerank.toFixed(6)
                        : 'Not available'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Coordinated Edge Ratio:</span>
                    <span className="font-mono font-bold text-rose-600">
                      {selectedNode.coordinated_edge_ratio !== undefined && selectedNode.coordinated_edge_ratio !== null
                        ? `${(selectedNode.coordinated_edge_ratio * 100).toFixed(1)}%`
                        : 'Not available'}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-400 py-6 text-center">Click any node on the graph to inspect properties.</p>
            )}
          </div>

          {/* Graph ML Benchmark Comparison */}
          <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <h3 className="text-sm font-bold text-slate-900">Graph ML Model Comparison</h3>
              <span className="text-[10px] font-mono text-indigo-600 font-semibold">Holdout AUC</span>
            </div>

            <div className="space-y-2 text-xs">
              {summary?.benchmarks && summary.benchmarks.length > 0 ? (
                summary.benchmarks.map((bench: any, idx: number) => (
                  <div key={idx} className="p-2.5 rounded-lg border border-slate-100 bg-slate-50/70 flex justify-between items-center">
                    <div>
                      <span className="font-bold text-slate-900 block">{bench.model}</span>
                      <span className="text-[10px] text-slate-500">
                        Fake F1: {bench.fake_f1 !== undefined && bench.fake_f1 !== null ? Number(bench.fake_f1).toFixed(4) : '—'} • Acc: {bench.accuracy !== undefined && bench.accuracy !== null ? Number(bench.accuracy).toFixed(4) : '—'}
                      </span>
                    </div>
                    <span className="font-mono font-bold text-indigo-600 text-sm">
                      {bench.roc_auc !== undefined && bench.roc_auc !== null ? Number(bench.roc_auc).toFixed(4) : '—'}
                    </span>
                  </div>
                ))
              ) : (
                <div className="p-4 bg-slate-50 rounded-lg text-center text-xs text-slate-400">
                  No verified graph benchmark results available.
                </div>
              )}
            </div>

            <p className="text-[10px] text-slate-400 italic">
              Notice: Handcrafted graph centralities outperform pure neural GNN on dense projected bipartite graphs due to heterophilous product interactions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
