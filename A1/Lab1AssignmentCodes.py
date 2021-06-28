import networkx as nx
import matplotlib.pylab as plt

G = nx.Graph()

# Add nodes
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_node(6)
G.add_node(7)
G.add_node(8)

# Add edge
G.add_edges_from([(1, 5), (2, 6), (3, 7)], weight=50)
G.add_edges_from([(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 8), (7, 8)], weight=40)
G.add_edges_from([(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 8)], weight=30)

nx.draw(G, with_labels=True, width=1)

print(nx. shortest_path (G, 1, 8, weight='weight'))
print(nx .shortest_path_length(G, 1, 8, weight='weight'))

plt.show()
