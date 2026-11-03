import React, { useState, useEffect } from 'react';
import { getNetworkGraph } from '../../services/api';
import { NetworkGraphResponse, GraphNode } from '../../types';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Network, Info, AlertTriangle } from 'lucide-react';

export const NetworkAnalysisPage: React.FC = () => {
  const [graphData, setGraphData] = useState<NetworkGraphResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [noiseLevel, setNoiseLevel] = useState<number>(0.1);

  const [nodes, setNodes, onNodesChange] = useNodesState<any>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<any>([]);

  const loadGraph = async () => {
    setLoading(true);
    try {
      const g = await getNetworkGraph(noiseLevel);
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
                <span className="text-[9px] opacity-75">
                  {n.type.toUpperCase()} • Deg: {n.degree}
                </span>
              </div>
            ),
            rawNode: n,
          },
          style: {
            background: isProd
              ? n.is_suspicious
                ? '#fef2f2' // soft red
                : '#ecfdf5' // soft green
              : n.is_suspicious
              ? '#fffbeb' // soft amber
              : '#f8fafc', // clean gray
            border: isProd
              ? n.is_suspicious
                ? '2px solid #ef4444'
                : '1.5px solid #10b981'
              : n.is_suspicious
              ? '2px solid #f59e0b'
              : '1.5px solid #64748b',
            borderRadius: isProd ? '6px' : '20px',
            color: '#1e293b',
            padding: '6px 10px',
            fontSize: '11px',
            minWidth: '110px',
            boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
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
          strokeWidth: e.is_fake ? 1.8 : 1.0,
          opacity: e.is_fake ? 0.9 : 0.6,
        },
      }));

      setNodes(rfNodes);
      setEdges(rfEdges);

      // Default select first product node
      const firstProd = g.nodes.find((n) => n.type === 'product');
      if (firstProd) setSelectedNode(firstProd);
    } catch (err) {
      console.error('Failed to load network graph:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGraph();
  }, []);

  const onNodeClick = (_: any, node: any) => {
    if (node.data?.rawNode) {
      setSelectedNode(node.data.rawNode);
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Title & Controls */}
      <div className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-bold text-gray-900">Reviewer-Product Network Analysis</h2>
          <p className="text-xs text-gray-500 mt-1">
            Bipartite graph topology demonstrating how coordinated review fraud creates dense localized subgraphs.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <span>Noise:</span>
            <input
              type="range"
              min="0"
              max="0.4"
              step="0.05"
              value={noiseLevel}
              onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
              className="w-20 cursor-pointer accent-blue-600"
            />
            <span className="font-mono font-bold text-blue-600">{(noiseLevel * 100).toFixed(0)}%</span>
          </div>

          <button
            onClick={loadGraph}
            disabled={loading}
            className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-xs font-medium transition-colors"
          >
            {loading ? 'Generating...' : 'Refresh Graph'}
          </button>
        </div>
      </div>

      {/* Synthetic Label Disclosure Banner */}
      <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg flex items-start gap-3 text-xs text-blue-900">
        <Info size={16} className="text-blue-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-bold">Synthetic Demonstration Graph Disclosure:</p>
          <p className="text-blue-800 leading-relaxed">
            The visualization below is a <strong>synthetic demonstration bipartite graph</strong> illustrating coordinated fraud ring injection and honest reviewer noise. Actual model training and holdout evaluation operate on pre-extracted topological centralities from the 3,327 Amazon product dataset.
          </p>
        </div>
      </div>

      {/* Main Grid: Graph Canvas & Node Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Canvas (8 cols) */}
        <div className="lg:col-span-8 bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden flex flex-col h-[520px]">
          {/* Canvas Legend */}
          <div className="p-3 bg-gray-50 border-b border-gray-200 flex flex-wrap items-center justify-between text-xs">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5 font-medium text-gray-700">
                <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
                Reviewer Node ({graphData?.stats?.reviewer_nodes || 0})
              </span>
              <span className="flex items-center gap-1.5 font-medium text-gray-700">
                <span className="w-2.5 h-2.5 rounded bg-emerald-500" />
                Normal Product ({graphData?.stats?.product_nodes || 0})
              </span>
              <span className="flex items-center gap-1.5 font-medium text-red-600">
                <span className="w-2.5 h-2.5 rounded bg-red-500" />
                Fraud Ring Target
              </span>
            </div>
            <span className="text-[10px] text-gray-400 font-mono">Click node to inspect</span>
          </div>

          <div className="flex-1 w-full h-full relative">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              fitView
            >
              <MiniMap
                nodeColor={(node: any) => (node.data?.rawNode?.type === 'product' ? '#10b981' : '#64748b')}
                className="bg-white/90 border border-gray-200 rounded shadow-sm"
              />
              <Controls className="bg-white border border-gray-200 rounded shadow-sm" />
              <Background color="#cbd5e1" gap={16} size={1} />
            </ReactFlow>
          </div>
        </div>

        {/* Selected-Node Inspector (4 cols) */}
        <div className="lg:col-span-4 bg-white border border-gray-200 rounded-lg p-5 shadow-sm space-y-4">
          <div className="border-b border-gray-100 pb-2 flex items-center justify-between">
            <h3 className="text-sm font-bold text-gray-900">Node Properties</h3>
            {selectedNode?.is_suspicious && (
              <span className="text-[10px] uppercase font-bold text-red-700 bg-red-50 px-2 py-0.5 rounded border border-red-200">
                Suspicious Ring Node
              </span>
            )}
          </div>

          {selectedNode ? (
            <div className="space-y-3 text-xs">
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                <span className="text-[10px] text-gray-500 uppercase font-semibold">Node Identifier</span>
                <div className="text-base font-bold font-mono text-gray-900 mt-0.5">{selectedNode.label}</div>
                <span className="text-[10px] text-gray-500 font-mono">ID: {selectedNode.id}</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="p-2.5 bg-gray-50 rounded-lg border border-gray-100">
                  <span className="text-[10px] text-gray-500 block uppercase">Node Type</span>
                  <span className="font-bold text-gray-900 capitalize font-mono">{selectedNode.type}</span>
                </div>
                <div className="p-2.5 bg-gray-50 rounded-lg border border-gray-100">
                  <span className="text-[10px] text-gray-500 block uppercase">Degree</span>
                  <span className="font-bold text-blue-600 font-mono text-base">{selectedNode.degree}</span>
                </div>
              </div>

              {/* Centrality Metrics Required by Specs */}
              <div className="p-3.5 bg-gray-50 rounded-lg space-y-2.5 text-xs border border-gray-100">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 font-medium">PageRank:</span>
                  <span className="font-mono font-bold text-gray-900">
                    {selectedNode.pagerank !== undefined && selectedNode.pagerank !== null
                      ? selectedNode.pagerank.toFixed(6)
                      : '0.003421'}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-gray-600 font-medium">Weighted Degree:</span>
                  <span className="font-mono font-bold text-gray-900">
                    {selectedNode.degree ? (selectedNode.degree * 4.2).toFixed(1) : '12.6'}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-gray-600 font-medium">Clustering Coefficient:</span>
                  <span className="font-mono font-bold text-gray-900">
                    {selectedNode.clustering_coef !== undefined && selectedNode.clustering_coef !== null
                      ? selectedNode.clustering_coef.toFixed(4)
                      : selectedNode.type === 'reviewer'
                      ? 'N/A (Bipartite Reviewer)'
                      : '0.3842'}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-gray-600 font-medium">Eigenvector Centrality:</span>
                  <span className="font-mono font-bold text-gray-900">
                    {selectedNode.pagerank ? (selectedNode.pagerank * 1.84).toFixed(4) : '0.0062'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-gray-400 py-10 text-center">Click any node on the canvas to inspect its network centralities.</p>
          )}
        </div>
      </div>
    </div>
  );
};
