import igraph
import plotly.graph_objs as go


class InteractiveGraph:
    """Interactive tree graph.

    Args:
        filename: Name of the .gml file.

    Attributes:
        G: Graph loaded from gml file.
        edges: Edges in the graph.
        ids: Indexes of nodes in graph.
        labels: Labels of nodes from gml file.
        num_nodes: Number of nodes.
        tree_height: Height of the graph.
        positions: Positions of nodes in graph.
        lines: Representation of edges.
        dots: Representation of nodes.
    """

    def __init__(self, filename):

        self.G = igraph.Graph.Read_GML(filename)
        self.edges = [e.tuple for e in self.G.es]
        self.ids = self.G.vs['id']
        self.labels = self.G.vs['label']
        self.num_nodes = len(self.ids)
        self.tree_height = 1

        self.positions = self._make_positions()
        self.lines = self._make_lines()
        self.dots = self._make_dots()

    def _make_positions(self):

        kids = [edge[1] for edge in self.edges if edge[0] == 0]

        positions = {0: [0, 0]}

        while kids:

            width_range = (len(kids) - 1.) / 2
            for i, kid in enumerate(kids):
                positions[kid] = [-width_range + i, self.tree_height]

            parents = kids
            kids = [edge[1] for parent in parents for edge in self.edges if edge[0] == parent]
            self.tree_height += 1

        return positions

    def _nodes_coords(self):

        Xn = [self.positions[k][0] for k in range(self.num_nodes)]
        Yn = [2 * self.tree_height - self.positions[k][1] for k in range(self.num_nodes)]

        return Xn, Yn

    def _edges_coords(self):

        Xe = []
        Ye = []
        for edge in self.edges:
            Xe += [self.positions[edge[0]][0], self.positions[edge[1]][0], None]
            Ye += [2 * self.tree_height - self.positions[edge[0]][1],
                   2 * self.tree_height - self.positions[edge[1]][1], None]

        return Xe, Ye

    def _make_dots(self):

        Xn, Yn = self._nodes_coords()

        dots = go.Scatter(x=Xn,
                          y=Yn,
                          mode='markers',
                          name='',
                          marker=dict(symbol='dot',
                                      size=40,
                                      color='#6175c1',
                                      line=dict(color='rgb(50,50,50)', width=1, cauto=False)
                                      ),
                          text=self.labels,
                          hoverinfo='text',
                          opacity=0.8
                          )
        return dots

    def _make_lines(self):

        Xe, Ye = self._edges_coords()

        lines = go.Scatter(x=Xe,
                           y=Ye,
                           mode='lines',
                           line=dict(color='rgb(210,210,210)', width=1),
                           hoverinfo='none'
                           )
        return lines

    def make_annotations(self, labels=[], font_size=10, font_color='rgb(250,250,250)'):
        """Interactive tree graph.

        Args:
            labels: New labels.
            font_size: Size of the font.
            font_color: Color of the font.

        Returns:
            annotations: Graph annotation.
        """

        if not labels:
            labels = self.labels

        elif len(labels) != self.num_nodes:
            raise ValueError('The lists positions and labels must have the same len')

        annotations = go.Annotations()
        for k in range(self.num_nodes):
            annotations.append(
                go.Annotation(
                    text=labels[k],
                    x=self.positions[k][0], y=2*self.tree_height-self.positions[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False
                )
            )
        return annotations


def prepare_graph(filename):
    """Prepare a graph to plot.

    Args:
        filename: Name of the .gml file.

    Returns:
        fig : The figure prepare for plotly.iplot function.
    """

    G = InteractiveGraph(filename)

    axis = dict(showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )

    layout = dict(annotations=G.make_annotations(),
                  font=dict(size=15),
                  showlegend=False,
                  xaxis=go.XAxis(axis),
                  yaxis=go.YAxis(axis),
                  margin=dict(l=10, r=10, b=15, t=15),
                  hovermode='closest',
                  plot_bgcolor='rgb(248,248,248)'
                  )

    data = go.Data([G.lines, G.dots])
    fig = dict(data=data, layout=layout)
    fig['layout'].update(annotations=G.make_annotations(G.ids, 15))

    return fig
