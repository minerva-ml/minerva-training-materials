# -*- coding: utf-8 -*-
import re

import pydot_ng as pydot
from IPython.display import Image, display


def view_pydot(pydot_object):
    plt = Image(pydot_object.create_png())
    display(plt)


def create_graph(graph_info):
    dot = pydot.Dot()
    for node in graph_info['nodes']:
        dot.add_node(pydot.Node(node))
    for node1, node2 in graph_info['edges']:
        dot.add_edge(pydot.Edge(node1, node2))
    return dot


def view_graph(graph_info):
    graph = create_graph(graph_info)
    view_pydot(graph)


def plot_graph(graph_info, filepath):
    graph = create_graph(graph_info)
    graph.write(filepath, format='png')


def get_unique_channel_name(ctx, basename, *, suffix=None, force_id=False):
    """
    Get unique channel name for given Neptune context
    :param ctx: neptune context already containing channels
    :param basename: base name (ex. batch_loss)
    :param suffix: suffix to identify experiment step
    :param force_id: force id even when no
    :return: unique for given Neptune context channel name
    """
    name = basename
    if suffix:
        name += ' ({})'.format(suffix)
    channels_with_name = [channel for channel in ctx._experiment._channels if name in channel.name]
    if not channels_with_name and not force_id:
        return name
    else:
        # obtain numeric suffix from channel name or '0' if filtered channels list is empty
        channel_ids = [re.split('[^\d]', channel.name)[-1] for channel in channels_with_name] or ['0']
        channel_ids = sorted([int(idx) if idx != '' else 0 for idx in channel_ids])
        last_id = channel_ids[-1]

    corrected_name = '{} {}'.format(name, last_id + 1)
    return corrected_name
