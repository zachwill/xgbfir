#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Xgbfir is a XGBoost model dump parser, which ranks features as well as
# feature interactions by different metrics.
# Copyright (c) 2016 Boris Kostenko
# https://github.com/limexp/xgbfir/
#
# Originally based on implementation by Far0n
# https://github.com/Far0n/xgbfi

# TODO: Should `division` be imported too?
from __future__ import print_function

import argparse
import sys
import re

import xlsxwriter


_COMPARER = None


def feature_score_comparer(metric):
    global _COMPARER
    _COMPARER = {
        'gain': lambda x: -x.gain,
        'fscore': lambda x: -x.FScore,
        'wfscore': lambda x: -x.wfscore,
        'average_wfscore': lambda x: -x.average_wfscore,
        'averagegain': lambda x: -x.average_gain,
        'expectedgain': lambda x: -x.expected_gain,
    }[metric.lower()]


class SplitValueHistogram:
    def __init__(self):
        self.values = {}

    def add_value(self, split_value, count):
        if not (split_value in self.values):
            self.values[split_value] = 0
        self.values[split_value] += count

    def merge(self, histogram):
        for key, value in histogram.values.items():
            self.add_value(key, value)


class FeatureInteraction:
    def __init__(self, interaction, gain, cover, path_probability, depth, tree_index, fscore=1):
        self.histogram = SplitValueHistogram()

        features = sorted(interaction, key=lambda x: x.feature)
        self.name = "|".join(x.feature for x in features)

        self.depth = len(interaction) - 1
        self.gain = gain
        self.cover = cover
        self.fscore = fscore
        self.wfscore = path_probability
        self.tree_index = tree_index
        self.tree_depth = depth
        # TODO: Might need to set `expected_gain`
        self.has_leaf_stats = False

        if self.depth == 0:
            self.histogram.add_value(interaction[0].split_value, 1)

        self.sum_leaf_values_left = 0.0
        self.sum_leaf_values_right = 0.0

        self.sum_leaf_covers_left = 0.0
        self.sum_leaf_covers_right = 0.0

    @property
    def average_wfscore(self):
        return self.wfscore / self.fscore

    @property
    def average_gain(self):
        return self.gain / self.fscore

    @property
    def expected_gain(self):
        # TODO: This could be wrong -- it was originally a += stat
        return self.gain * self.wfscore

    @property
    def average_tree_index(self):
        return self.tree_index / self.fscore

    @property
    def average_tree_depth(self):
        return self.tree_depth / self.fscore

    def __lt__(self, other):
        return self.name < other.name


class FeatureInteractions:
    def __init__(self):
        self.interactions = {}
        # TODO: Shouldn't comparsion take place here? And be an initial param?

    @property
    def count(self):
        # TODO: This was never used in previous code
        return len(self.interactions.keys())

    def interactions_of_depth(self, depth):
        return sorted([
            self.interactions[key] for key in self.interactions.keys() if self.interactions[key].depth == depth],
            key=_COMPARER
        )

    def interactions_with_leaf_stats(self):
        return sorted([
            self.interactions[key] for key in self.interactions.keys() if self.interactions[key].has_leaf_stats],
            key=_COMPARER
        )

    def merge(self, other):
        for key in other.interactions.keys():
            feature = other.interactions[key]
            if key not in self.interactions:
                self.interactions[key] = feature
            else:
                self.interactions[key].gain += feature.gain
                self.interactions[key].cover += feature.cover
                self.interactions[key].fscore += feature.fscore
                self.interactions[key].wfscore += feature.wfscore
                self.interactions[key].tree_index += feature.tree_index
                self.interactions[key].tree_depth += feature.tree_depth
                self.interactions[key].sum_leaf_covers_left += feature.sum_leaf_covers_left
                self.interactions[key].sum_leaf_covers_right += feature.sum_leaf_covers_right
                self.interactions[key].sum_leaf_values_left += feature.sum_leaf_values_left
                self.interactions[key].sum_leaf_values_right += feature.sum_leaf_values_right
                self.interactions[key].histogram.merge(feature.histogram)


class XGBModel:
    def __init__(self, verbosity=0):
        self.xgb_trees = []
        self._verbosity = verbosity
        self._tree_index = 0
        self._maxDeepening = 0
        self._pathMemo = []
        self._maxInteractionDepth = 0

    def add_tree(self, tree):
        self.xgb_trees.append(tree)

    def feature_interactions(self, maxInteractionDepth, maxDeepening):
        xgbFeatureInteractions = FeatureInteractions()
        self._maxInteractionDepth = maxInteractionDepth
        self._maxDeepening = maxDeepening

        if self._verbosity >= 1:
            if self._maxInteractionDepth == -1:
                print("Collecting feature interactions")
            else:
                print("Collecting feature interactions up to depth {}".format(self._maxInteractionDepth))

        for i, tree in enumerate(self.xgb_trees):
            if self._verbosity >= 2:
                sys.stdout.write("Collecting feature interactions within tree #{} ".format(i + 1))

            self._treeFeatureInteractions = FeatureInteractions()
            self._pathMemo = []
            self._tree_index = i

            tree_nodes = []
            self.collect_interactions(tree, tree_nodes)

            if self._verbosity >= 2:
                sys.stdout.write("=> number of interactions: {}\n".format(len(self._treeFeatureInteractions.interactions)))
            xgbFeatureInteractions.merge(self._treeFeatureInteractions)

        if self._verbosity >= 1:
            print("{} feature interactions has been collected.".format(len(xgbFeatureInteractions.interactions)))

        return xgbFeatureInteractions

    def collect_interactions(self, tree, currentInteraction, currentGain=0.0, currentCover=0.0, path_probability=1.0, depth=0, deepening=0):
        if tree.node.is_leaf:
            return

        currentInteraction.append(tree.node)
        currentGain += tree.node.gain
        currentCover += tree.node.cover

        path_probability_left = path_probability * (tree.left.node.cover / tree.node.cover)
        path_probability_right = path_probability * (tree.right.node.cover / tree.node.cover)

        fi = FeatureInteraction(currentInteraction, currentGain, currentCover, path_probability, depth, self._tree_index, 1)

        if (depth < self._maxDeepening) or (self._maxDeepening < 0):
            newInteractionLeft = []
            newInteractionRight = []

            self.collect_interactions(tree.left, newInteractionLeft, 0.0, 0.0, path_probability_left, depth + 1, deepening + 1)
            self.collect_interactions(tree.right, newInteractionRight, 0.0, 0.0, path_probability_right, depth + 1, deepening + 1)

        path = ",".join(str(n.number) for n in currentInteraction)

        if fi.name not in self._treeFeatureInteractions.interactions:
            self._treeFeatureInteractions.interactions[fi.name] = fi
            self._pathMemo.append(path)
        else:
            if path in self._pathMemo:
                return
            self._pathMemo.append(path)

            tfi = self._treeFeatureInteractions.interactions[fi.name]
            tfi.gain += currentGain
            tfi.cover += currentCover
            tfi.fscore += 1
            tfi.wfscore += path_probability
            tfi.tree_depth += depth
            tfi.tree_index += self._tree_index
            tfi.histogram.merge(fi.histogram)

        if len(currentInteraction) - 1 == self._maxInteractionDepth:
            return

        currentInteractionLeft = list(currentInteraction)
        currentInteractionRight = list(currentInteraction)

        left_tree = tree.left
        right_tree = tree.right

        if left_tree.node.is_leaf and deepening == 0:
            tfi = self._treeFeatureInteractions.interactions[fi.name]
            tfi.sum_leaf_values_left += left_tree.node.leaf_value
            tfi.sum_leaf_covers_left += left_tree.node.cover
            tfi.has_leaf_stats = True

        if right_tree.node.is_leaf and deepening == 0:
            tfi = self._treeFeatureInteractions.interactions[fi.name]
            tfi.sum_leaf_values_right += right_tree.node.leaf_value
            tfi.sum_leaf_covers_right += right_tree.node.cover
            tfi.has_leaf_stats = True

        self.collect_interactions(tree.left, currentInteractionLeft, currentGain, currentCover, path_probability_left, depth + 1, deepening)
        self.collect_interactions(tree.right, currentInteractionRight, currentGain, currentCover, path_probability_right, depth + 1, deepening)


class XGBTreeNode:
    def __init__(self):
        self.feature = ''
        self.gain = 0.0
        self.cover = 0.0
        self.number = -1
        self.left_child = None
        self.right_child = None
        self.leaf_value = 0.0
        self.split_value = 0.0
        self.is_leaf = False

    def __lt__(self, other):
        return self.number < other.number


class XGBTree:
    def __init__(self, node):
        self.left = None
        self.right = None
        self.node = node  # or node.copy()


class XGBModelParser:
    def __init__(self, verbosity=0):
        self._verbosity = verbosity
        self.node_regex = re.compile("(\d+):\[(.*)<(.+)\]\syes=(.*),no=(.*),missing=.*,gain=(.*),cover=(.*)")
        self.leaf_regex = re.compile("(\d+):leaf=(.*),cover=(.*)")
        self.xgb_node_list = {}

    def construct_tree(self, tree):
        if tree.node.left_child is not None:
            tree.left = XGBTree(self.xgb_node_list[tree.node.left_child])
            self.construct_tree(tree.left)
        if tree.node.right_child is not None:
            tree.right = XGBTree(self.xgb_node_list[tree.node.right_child])
            self.construct_tree(tree.right)

    def parse_tree_node(self, line):
        node = XGBTreeNode()
        if "leaf" in line:
            m = self.leaf_regex.match(line)
            node.is_leaf = True
            node.number = int(m.group(1))
            node.leaf_value = float(m.group(2))
            node.cover = float(m.group(3))
        else:
            m = self.node_regex.match(line)
            node.is_leaf = False
            node.number = int(m.group(1))
            node.feature = m.group(2)
            node.split_value = float(m.group(3))
            node.left_child = int(m.group(4))
            node.right_child = int(m.group(5))
            node.gain = float(m.group(6))
            node.cover = float(m.group(7))
        return node

    def model_from_file(self, file_name, max_trees):
        model = XGBModel(self._verbosity)
        self.xgb_node_list = {}
        number_of_trees = 0
        with open(file_name) as f:
            for line in f:
                line = line.strip()
                if (not line) or line.startswith('booster'):
                    if any(self.xgb_node_list):
                        number_of_trees += 1
                        if self._verbosity >= 2:
                            sys.stdout.write("Constructing tree #{}\n".format(number_of_trees))
                        tree = XGBTree(self.xgb_node_list[0])
                        self.construct_tree(tree)

                        model.add_tree(tree)
                        self.xgb_node_list = {}
                        if number_of_trees == max_trees:
                            if self._verbosity >= 1:
                                print("Maximum number of trees reached: #{}".format(max_trees))
                            break
                else:
                    node = self.parse_tree_node(line)
                    if not node:
                        return None
                    self.xgb_node_list[node.number] = node

            if any(self.xgb_node_list) and ((max_trees < 0) or (number_of_trees < max_trees)):
                number_of_trees += 1
                if self._verbosity >= 2:
                    sys.stdout.write("Constructing tree #{}\n".format(number_of_trees))
                tree = XGBTree(self.xgb_node_list[0])
                self.construct_tree(tree)

                model.add_tree(tree)
                self.xgb_node_list = {}

        return model

    def model_from_memory(self, dump, max_trees):
        model = XGBModel(self._verbosity)
        self.xgb_node_list = {}
        number_of_trees = 0
        for booster_line in dump:
            self.xgb_node_list = {}
            for line in booster_line.split('\n'):
                line = line.strip()
                if not line:
                    continue
                node = self.parse_tree_node(line)
                if not node:
                    return None
                self.xgb_node_list[node.number] = node
            number_of_trees += 1
            tree = XGBTree(self.xgb_node_list[0])
            self.construct_tree(tree)
            model.add_tree(tree)
            if number_of_trees == max_trees:
                break
        return model


def rank_inplace(a):
    c = [(j, i[0]) for j, i in enumerate(sorted(enumerate(a), key=lambda x:x[1]))]
    c.sort(key=lambda x: x[1])
    return [i[0] for i in c]


def FeatureInteractionsWriter(FeatureInteractions, file_name, MaxDepth, topK, MaxHistograms, verbosity=0):

    if verbosity >= 1:
        print("Writing {}".format(file_name))

    workbook = xlsxwriter.Workbook(file_name)

    cf_first_row = workbook.add_format()
    cf_first_row.set_align('center')
    cf_first_row.set_align('vcenter')
    cf_first_row.set_bold(True)

    cf_first_column = workbook.add_format()
    cf_first_column.set_align('center')
    cf_first_column.set_align('vcenter')

    cf_num = workbook.add_format()
    cf_num.set_num_format('0.00')

    for depth in range(MaxDepth + 1):
        if verbosity >= 1:
            print("Writing feature interactions with depth {}".format(depth))

        interactions = FeatureInteractions.interactions_of_depth(depth)

        KTotalGain = sum([i.gain for i in interactions])
        TotalCover = sum([i.cover for i in interactions])
        TotalFScore = sum([i.fscore for i in interactions])
        TotalFScoreWeighted = sum([i.wfscore for i in interactions])
        TotalFScoreWeightedAverage = sum([i.average_wfscore for i in interactions])

        if topK > 0:
            interactions = interactions[0:topK]

        if not interactions:
            break

        ws = workbook.add_worksheet("Interaction Depth {}".format(depth))

        ws.set_row(0, 20, cf_first_row)

        ws.set_column(0, 0, max([len(i.name) for i in interactions]) + 10, cf_first_column)

        ws.set_column(1, 13, 17)
        ws.set_column(10, 11, 18)
        ws.set_column(12, 12, 19)
        ws.set_column(13, 13, 17, cf_num)
        ws.set_column(14, 15, 19, cf_num)

        for col, name in enumerate([
            "Interaction", "Gain", "FScore", "wFScore", "Average wFScore", "Average Gain", "Expected Gain",
            "Gain Rank", "FScore Rank", "wFScore Rank", "Avg wFScore Rank", "Avg Gain Rank", "Expected Gain Rank",
            "Average Rank", "Average Tree Index", "Average Tree Depth"
        ]):
            ws.write(0, col, name)

        gainSorted = rank_inplace([-f.gain for f in interactions])
        fScoreSorted = rank_inplace([-f.fscore for f in interactions])
        fScoreWeightedSorted = rank_inplace([-f.wfscore for f in interactions])
        averagefScoreWeightedSorted = rank_inplace([-f.average_wfscore for f in interactions])
        averageGainSorted = rank_inplace([-f.average_gain for f in interactions])
        expectedGainSorted = rank_inplace([-f.expected_gain for f in interactions])

        for i, fi in enumerate(interactions):
            ws.write(i + 1, 0, fi.name)
            ws.write(i + 1, 1, fi.gain)
            ws.write(i + 1, 2, fi.fscore)
            ws.write(i + 1, 3, fi.wfscore)
            ws.write(i + 1, 4, fi.average_wfscore)
            ws.write(i + 1, 5, fi.average_gain)
            ws.write(i + 1, 6, fi.expected_gain)
            ws.write(i + 1, 7, 1 + gainSorted[i])
            ws.write(i + 1, 8, 1 + fScoreSorted[i])
            ws.write(i + 1, 9, 1 + fScoreWeightedSorted[i])
            ws.write(i + 1, 10, 1 + averagefScoreWeightedSorted[i])
            ws.write(i + 1, 11, 1 + averageGainSorted[i])
            ws.write(i + 1, 12, 1 + expectedGainSorted[i])
            ws.write(i + 1, 13, (6.0 + gainSorted[i] + fScoreSorted[i] + fScoreWeightedSorted[i] + averagefScoreWeightedSorted[i] + averageGainSorted[i] + expectedGainSorted[i]) / 6.0)
            ws.write(i + 1, 14, fi.average_tree_index)
            ws.write(i + 1, 15, fi.average_tree_depth)

    interactions = FeatureInteractions.interactions_with_leaf_stats()
    if interactions:
        if verbosity >= 1:
            print("Writing leaf statistics")

        ws = workbook.add_worksheet("Leaf Statistics")

        ws.set_row(0, 20, cf_first_row)
        ws.set_column(0, 0, max([len(i.name) for i in interactions]) + 10, cf_first_column)
        ws.set_column(1, 4, 20)

        for col, name in enumerate([
            "Interaction", "Sum Leaf Values Left", "Sum Leaf Values Right", "Sum Leaf Covers Left", "Sum Leaf Covers Right"
        ]):
            ws.write(0, col, name)

        for i, fi in enumerate(interactions):
            ws.write(i + 1, 0, fi.name)
            ws.write(i + 1, 1, fi.sum_leaf_values_left)
            ws.write(i + 1, 2, fi.sum_leaf_values_right)
            ws.write(i + 1, 3, fi.sum_leaf_covers_left)
            ws.write(i + 1, 4, fi.sum_leaf_covers_right)

    interactions = FeatureInteractions.interactions_of_depth(0)
    if interactions:
        if verbosity >= 1:
            print("Writing split value histograms")

        ws = workbook.add_worksheet("Split Value Histograms")

        ws.set_row(0, 20, cf_first_row)
        ws.set_column(0, 0, max([len(i.name) for i in interactions]) + 10, cf_first_column)
        ws.set_column(1, 4, 20)

        for col, name in enumerate([
            "Interaction", "Sum Leaf Values Left", "Sum Leaf Values Right", "Sum Leaf Covers Left", "Sum Leaf Covers Right"
        ]):
            ws.write(0, col, name)

        for i, fi in enumerate(interactions):
            if i >= MaxHistograms:
                break

            c1 = i * 2
            c2 = c1 + 1

            ws.merge_range(0, c1, 0, c2, fi.name)
            ws.set_column(c1, c1, max(10, (len(fi.name) + 4) / 2))
            ws.set_column(c2, c2, max(10, (len(fi.name) + 4) / 2))

            for j, key in enumerate(sorted(fi.histogram.values.keys())):
                ws.write(j + 1, c1, key)
                ws.write(j + 1, c2, fi.histogram.values[key])

    workbook.close()


def main(argv):
    epilog = '''
XGBoost Feature Interactions Reshaped 0.2
URL: https://github.com/limexp/xgbfir
'''

    arg_parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='XGBoost model dump parser, which ranks features as well as feature interactions by different metrics.',
        epilog=epilog)

    arg_parser.add_argument(
        '-V', '--version', action='version',
        version='XGBoost Feature Interactions Reshaped 0.2')

    arg_parser.add_argument(
        '-m', dest='XGBModelFile', action='store', default='xgb.dump',
        help="Xgboost model dump (dumped w/ 'with_stats=True')")

    arg_parser.add_argument(
        '-o', dest='output', action='store', default='XgbFeatureInteractions.xlsx',
        help='Xlsx file to be written')

    arg_parser.add_argument(
        '-t', dest='max_trees', action='store', default='100', type=int,
        help='Upper bound for trees to be parsed')

    arg_parser.add_argument(
        '-d', dest='MaxInteractionDepth', action='store', default='2', type=int,
        help='Upper bound for extracted feature interactions depth')

    arg_parser.add_argument(
        '-g', dest='MaxDeepening', action='store', default='-1', type=int,
        help='Upper bound for interaction start deepening (zero deepening => interactions starting @root only)')

    arg_parser.add_argument(
        '-k', dest='TopK', action='store', default='100', type=int,
        help='Upper bound for exported feature interactions per depth level')

    arg_parser.add_argument(
        '-H', dest='MaxHistograms', action='store', default='10', type=int,
        help='Maximum number of histograms')

    arg_parser.add_argument(
        '-s', dest='SortBy', action='store', default='Gain',
        help='Score metric to sort by (Gain, FScore, wFScore, AvgwFScore, AvgGain, ExpGain)')

    arg_parser.add_argument(
        '-v', '--verbosity', dest='Verbosity', action='count', default='2',
        help='Increate output verbosity')

    args = arg_parser.parse_args(args=argv[1:])

    args.XGBModelFile = args.XGBModelFile.strip()
    args.output = args.output.strip()

    verbosity = int(args.Verbosity)

    settings_print = '''
Settings:
=========
XGBModelFile (-m): {model}
output (-o): {output}
MaxInteractionDepth: {depth}
MaxDeepening (-g): {deepening}
max_trees (-t): {trees}
TopK (-k): {topk}
SortBy (-s): {sortby}
MaxHistograms (-H): {histograms}
'''.format(
        model=args.XGBModelFile,
        output=args.output,
        depth=args.MaxInteractionDepth,
        deepening=args.MaxDeepening,
        trees=args.max_trees,
        topk=args.TopK,
        sortby=args.SortBy,
        histograms=args.MaxHistograms
    )

    if verbosity >= 1:
        print(settings_print)

    feature_score_comparer(args.SortBy)

    parser = XGBModelParser(verbosity)
    model = parser.model_from_file(args.XGBModelFile, args.max_trees)
    interactions = model.feature_interactions(args.MaxInteractionDepth, args.MaxDeepening)

    FeatureInteractionsWriter(interactions, args.output, args.MaxInteractionDepth, args.TopK, args.MaxHistograms)

    if verbosity >= 1:
        print(epilog)

    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


def save_excel(booster, feature_names=None, output='XGBFeatureInteractions.xlsx', max_trees=100, MaxInteractionDepth=2, MaxDeepening=-1, TopK=100, MaxHistograms=10, SortBy='Gain'):
    if 'get_dump' not in dir(booster):
        if 'booster' in dir(booster):
            booster = booster.booster()
        else:
            # Shouldn't this be an exception?
            return -20
    if feature_names is not None:
        if isinstance(feature_names, list):
            booster.feature_names = feature_names
        else:
            booster.feature_names = list(feature_names)
    feature_score_comparer(SortBy)
    parser = XGBModelParser()
    dump = booster.get_dump('', with_stats=True)
    model = parser.model_from_memory(dump, max_trees)
    interactions = model.feature_interactions(MaxInteractionDepth, MaxDeepening)
    FeatureInteractionsWriter(interactions, output, MaxInteractionDepth, TopK, MaxHistograms)


if __name__ == '__main__':
    entry_point()
