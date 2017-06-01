# -*- coding: utf-8 -*-
from Queue import Queue
import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx


class Graph(object):
    """A directed graph."""
    def __init__(self, V, E):
        self.V = V
        self.E = E

    """Find a path with positive capacity"""
    def flow_path(self, source, dest):
        q = Queue()
        q.put(source)
        current = source
        visited = [source]

        while not q.empty():
            current = q.get()

            if current == dest:
                # Reconstruct the path backtracking through the parents.
                path = []
                while current.parent:
                    path.append(current.parent)
                    current = current.parent.source

                return Path(path[::-1])

            for e in current.outward_edges():
                if e.capacity > 0 and e.dest not in visited:
                    current = e.dest
                    visited.append(current)
                    q.put(current)
                    current.parent = e

    """Plot the graph"""
    def plot(self, filename="plot.html"):
        edge_trace = Scatter(
            x=[],
            y=[],
            line=Line(width=1, color='#888'),
            mode='lines')

        edge_label_trace = Scatter(
            x=[],
            y=[],
            text=[],
            textposition='middle center',
            textfont=dict(color='white'),
            mode='markers+text',
            marker=Marker(size=25, color='#888'))

        flow_edge_trace = Scatter(
            x=[],
            y=[],
            line=Line(width=2, color='blue'),
            mode='lines')

        flow_label_trace = Scatter(
            x=[],
            y=[],
            text=[],
            textposition='middle center',
            textfont=dict(color='white'),
            mode='markers+text',
            marker=Marker(size=25, color='blue'))

        node_trace = Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition='center',
            marker=Marker(
                color='white',
                size=30,
                line=dict(width=2)))

        for edge in self.E:
            x0, y0 = edge.source.position
            x1, y1 = edge.dest.position

            x = 1.*(x0 + x1) / 2
            y = 1.*(y0 + y1) / 2

            if edge.flow != 0:
                flow_edge_trace['x'] += [x0, x1, None]
                flow_edge_trace['y'] += [y0, y1, None]

                flow_label_trace['x'].append(x)
                flow_label_trace['y'].append(y)
                flow_label_trace['text'].append(edge.flow)
            else:
                edge_trace['x'] += [x0, x1, None]
                edge_trace['y'] += [y0, y1, None]
                edge_label_trace['x'].append(x)
                edge_label_trace['y'].append(y)
                edge_label_trace['text'].append(edge.capacity)


        for node in self.V:
            x, y = node.position
            node_trace['x'].append(x)
            node_trace['y'].append(y)
            node_trace['text'].append(node.label)

        fig = Figure(data=Data([edge_trace, edge_label_trace, flow_edge_trace, flow_label_trace, node_trace]),
                     layout=Layout(
                        title='<br>Network flow',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

        py.plot(fig, filename=filename)


class Node(object):
    """A node of the graph"""
    def __init__(self, label, position=(0,0)):
        self.label = label
        self.position = position
        self.edges = []
        self.parent = None

    def add_edge(self, edge):
        self.edges.append(edge)

    def outward_edges(self):
        return self.edges


class Edge(object):
    """A directed edge"""
    def __init__(self, source, dest, capacity=1, flow=0):
        self.source = source
        self.dest = dest
        self.capacity = capacity
        self.flow = flow

        source.add_edge(self)

class BidirectionalEdge(Edge):
    """A bidirectional edge (two directed edges)"""
    def __init__(self, source, dest, capacity=1, flow=0):
        self.source = source
        self.dest = dest
        self.capacity = capacity

        source.add_edge(self)
        dest.add_edge(self)


class Path(object):
    """A path on the graph"""
    def __init__(self, edges=[]):
        self.edges = edges

    def nodes(self):
        nodes = []
        if len(self.edges) >= 1:
            nodes.append(self.edges[0].source)

        for edge in self.edges:
            nodes.append(edge.dest)

        return nodes

    def capacity(self):
        print(self.edges)
        return min([edge.capacity for edge in self.edges])
