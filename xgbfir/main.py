#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Xgbfir is a XGBoost model dump parser, which ranks features as well as
# feature interactions by different metrics.
# Copyright (c) 2016 Boris Kostenko
# https://github.com/limexp/xgbfir/
#
# Originally based on implementation by Far0n
# https://github.com/Far0n/xgbfi

from __future__ import print_function

import argparse
import sys
import re
import xlsxwriter


_COMPARER = None


def FeatureScoreComparer(sortingMetric):
    global _COMPARER
    _COMPARER = {
        'gain': lambda x: -x.Gain,
        'fscore': lambda x: -x.FScore,
        'fscoreweighted': lambda x: -x.FScoreWeighted,
        'fscoreweightedaverage': lambda x: -x.FScoreWeightedAverage,
        'averagegain': lambda x: -x.AverageGain,
        'expectedgain': lambda x: -x.ExpectedGain
    }[sortingMetric.lower()]


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
        self.SplitValueHistogram = SplitValueHistogram()

        features = sorted(interaction, key=lambda x: x.Feature)
        self.Name = "|".join(x.Feature for x in features)

        self.Depth = len(interaction) - 1
        self.Gain = gain
        self.Cover = cover
        self.FScore = fscore
        self.FScoreWeighted = path_probability

        self.AverageFScoreWeighted = self.FScoreWeighted / self.FScore
        self.AverageGain = self.Gain / self.FScore
        self.ExpectedGain = self.Gain * path_probability
        self.TreeIndex = tree_index
        self.TreeDepth = depth
        self.AverageTreeIndex = self.TreeIndex / self.FScore
        self.AverageTreeDepth = self.TreeDepth / self.FScore
        self.HasLeafStatistics = False

        if self.Depth == 0:
            self.SplitValueHistogram.add_value(interaction[0].SplitValue, 1)

        self.SumLeafValuesLeft = 0.0
        self.SumLeafCoversLeft = 0.0
        self.SumLeafValuesRight = 0.0
        self.SumLeafCoversRight = 0.0

    def __lt__(self, other):
        return self.Name < other.Name


class FeatureInteractions:
    def __init__(self):
        self.Count = 0
        self.interactions = {}

    def interactions_of_depth(self, depth):
        return sorted([self.interactions[key] for key in self.interactions.keys() if self.interactions[key].Depth == depth],
                      key=_COMPARER)

    def interactions_with_leaf_stats(self):
        return sorted([self.interactions[key] for key in self.interactions.keys() if self.interactions[key].HasLeafStatistics],
                      key=_COMPARER)

    def Merge(self, other):
        for key in other.interactions.keys():
            fi = other.interactions[key]
            if not (key in self.interactions):
                self.interactions[key] = fi
            else:
                self.interactions[key].Gain += fi.Gain
                self.interactions[key].Cover += fi.Cover
                self.interactions[key].FScore += fi.FScore
                self.interactions[key].FScoreWeighted += fi.FScoreWeighted
                self.interactions[key].AverageFScoreWeighted = self.interactions[key].FScoreWeighted / self.interactions[key].FScore
                self.interactions[key].AverageGain = self.interactions[key].Gain / self.interactions[key].FScore
                self.interactions[key].ExpectedGain += fi.ExpectedGain
                self.interactions[key].SumLeafCoversLeft += fi.SumLeafCoversLeft
                self.interactions[key].SumLeafCoversRight += fi.SumLeafCoversRight
                self.interactions[key].SumLeafValuesLeft += fi.SumLeafValuesLeft
                self.interactions[key].SumLeafValuesRight += fi.SumLeafValuesRight
                self.interactions[key].TreeIndex += fi.TreeIndex
                self.interactions[key].AverageTreeIndex = self.interactions[key].TreeIndex / self.interactions[key].FScore
                self.interactions[key].TreeDepth += fi.TreeDepth
                self.interactions[key].AverageTreeDepth = self.interactions[key].TreeDepth / self.interactions[key].FScore

                self.interactions[key].SplitValueHistogram.merge(fi.SplitValueHistogram)


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

    def GetFeatureInteractions(self, maxInteractionDepth, maxDeepening):
        xgbFeatureInteractions = FeatureInteractions()
        self._maxInteractionDepth = maxInteractionDepth
        self._maxDeepening = maxDeepening

        if self._verbosity >= 1:
            if self._maxInteractionDepth == -1:
                print("Collectiong feature interactions")
            else:
                print("Collectiong feature interactions up to depth {}".format(self._maxInteractionDepth))

        for i, tree in enumerate(self.xgb_trees):
            if self._verbosity >= 2:
                sys.stdout.write("Collecting feature interactions within tree #{} ".format(i + 1))

            self._treeFeatureInteractions = FeatureInteractions()
            self._pathMemo = []
            self._tree_index = i

            tree_nodes = []
            self.CollectFeatureInteractions(tree, tree_nodes)

            if self._verbosity >= 2:
                sys.stdout.write("=> number of interactions: {}\n".format(len(self._treeFeatureInteractions.interactions)))
            xgbFeatureInteractions.Merge(self._treeFeatureInteractions)

        if self._verbosity >= 1:
            print("{} feature interactions has been collected.".format(len(xgbFeatureInteractions.interactions)))

        return xgbFeatureInteractions

    def CollectFeatureInteractions(self, tree, currentInteraction, currentGain=0.0, currentCover=0.0, path_probability=1.0, depth=0, deepening=0):
        if tree.node.is_leaf:
            return

        currentInteraction.append(tree.node)
        currentGain += tree.node.Gain
        currentCover += tree.node.Cover

        path_probability_left = path_probability * (tree.left.node.Cover / tree.node.Cover)
        path_probability_right = path_probability * (tree.right.node.Cover / tree.node.Cover)

        fi = FeatureInteraction(currentInteraction, currentGain, currentCover, path_probability, depth, self._tree_index, 1)

        if (depth < self._maxDeepening) or (self._maxDeepening < 0):
            newInteractionLeft = []
            newInteractionRight = []

            self.CollectFeatureInteractions(tree.left, newInteractionLeft, 0.0, 0.0, path_probability_left, depth + 1, deepening + 1)
            self.CollectFeatureInteractions(tree.right, newInteractionRight, 0.0, 0.0, path_probability_right, depth + 1, deepening + 1)

        path = ",".join(str(n.Number) for n in currentInteraction)

        if not (fi.Name in self._treeFeatureInteractions.interactions):
            self._treeFeatureInteractions.interactions[fi.Name] = fi
            self._pathMemo.append(path)
        else:
            if path in self._pathMemo:
                return
            self._pathMemo.append(path)

            tfi = self._treeFeatureInteractions.interactions[fi.Name]
            tfi.Gain += currentGain
            tfi.Cover += currentCover
            tfi.FScore += 1
            tfi.FScoreWeighted += path_probability
            tfi.AverageFScoreWeighted = tfi.FScoreWeighted / tfi.FScore
            tfi.AverageGain = tfi.Gain / tfi.FScore
            tfi.ExpectedGain += currentGain * path_probability
            tfi.TreeDepth += depth
            tfi.AverageTreeDepth = tfi.TreeDepth / tfi.FScore
            tfi.TreeIndex += self._tree_index
            tfi.AverageTreeIndex = tfi.TreeIndex / tfi.FScore
            tfi.SplitValueHistogram.merge(fi.SplitValueHistogram)

        if len(currentInteraction) - 1 == self._maxInteractionDepth:
            return

        currentInteractionLeft = list(currentInteraction)
        currentInteractionRight = list(currentInteraction)

        left_tree = tree.left
        right_tree = tree.right

        if left_tree.node.is_leaf and deepening == 0:
            tfi = self._treeFeatureInteractions.interactions[fi.Name]
            tfi.SumLeafValuesLeft += left_tree.node.LeafValue
            tfi.SumLeafCoversLeft += left_tree.node.Cover
            tfi.HasLeafStatistics = True

        if right_tree.node.is_leaf and deepening == 0:
            tfi = self._treeFeatureInteractions.interactions[fi.Name]
            tfi.SumLeafValuesRight += right_tree.node.LeafValue
            tfi.SumLeafCoversRight += right_tree.node.Cover
            tfi.HasLeafStatistics = True

        self.CollectFeatureInteractions(tree.left, currentInteractionLeft, currentGain, currentCover, path_probability_left, depth + 1, deepening)
        self.CollectFeatureInteractions(tree.right, currentInteractionRight, currentGain, currentCover, path_probability_right, depth + 1, deepening)


class XGBTreeNode:
    def __init__(self):
        self.Feature = ''
        self.Gain = 0.0
        self.Cover = 0.0
        self.Number = -1
        self.left_child = None
        self.right_child = None
        self.LeafValue = 0.0
        self.SplitValue = 0.0
        self.is_leaf = False

    def __lt__(self, other):
        return self.Number < other.Number


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
            node.Number = int(m.group(1))
            node.LeafValue = float(m.group(2))
            node.Cover = float(m.group(3))
            node.is_leaf = True
        else:
            m = self.node_regex.match(line)
            node.Number = int(m.group(1))
            node.Feature = m.group(2)
            node.SplitValue = float(m.group(3))
            node.left_child = int(m.group(4))
            node.right_child = int(m.group(5))
            node.Gain = float(m.group(6))
            node.Cover = float(m.group(7))
            node.is_leaf = False
        return node

    def model_from_file(self, fileName, max_trees):
        model = XGBModel(self._verbosity)
        self.xgb_node_list = {}
        number_of_trees = 0
        with open(fileName) as f:
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
                    self.xgb_node_list[node.Number] = node

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
                self.xgb_node_list[node.Number] = node
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


def FeatureInteractionsWriter(FeatureInteractions, fileName, MaxDepth, topK, MaxHistograms, verbosity=0):

    if verbosity >= 1:
        print("Writing {}".format(fileName))

    workbook = xlsxwriter.Workbook(fileName)

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

        KTotalGain = sum([i.Gain for i in interactions])
        TotalCover = sum([i.Cover for i in interactions])
        TotalFScore = sum([i.FScore for i in interactions])
        TotalFScoreWeighted = sum([i.FScoreWeighted for i in interactions])
        TotalFScoreWeightedAverage = sum([i.AverageFScoreWeighted for i in interactions])

        if topK > 0:
            interactions = interactions[0:topK]

        if not interactions:
            break

        ws = workbook.add_worksheet("Interaction Depth {}".format(depth))

        ws.set_row(0, 20, cf_first_row)

        ws.set_column(0, 0, max([len(i.Name) for i in interactions]) + 10, cf_first_column)

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

        gainSorted = rank_inplace([-f.Gain for f in interactions])
        fScoreSorted = rank_inplace([-f.FScore for f in interactions])
        fScoreWeightedSorted = rank_inplace([-f.FScoreWeighted for f in interactions])
        averagefScoreWeightedSorted = rank_inplace([-f.AverageFScoreWeighted for f in interactions])
        averageGainSorted = rank_inplace([-f.AverageGain for f in interactions])
        expectedGainSorted = rank_inplace([-f.ExpectedGain for f in interactions])

        for i, fi in enumerate(interactions):
            ws.write(i + 1, 0, fi.Name)
            ws.write(i + 1, 1, fi.Gain)
            ws.write(i + 1, 2, fi.FScore)
            ws.write(i + 1, 3, fi.FScoreWeighted)
            ws.write(i + 1, 4, fi.AverageFScoreWeighted)
            ws.write(i + 1, 5, fi.AverageGain)
            ws.write(i + 1, 6, fi.ExpectedGain)
            ws.write(i + 1, 7, 1 + gainSorted[i])
            ws.write(i + 1, 8, 1 + fScoreSorted[i])
            ws.write(i + 1, 9, 1 + fScoreWeightedSorted[i])
            ws.write(i + 1, 10, 1 + averagefScoreWeightedSorted[i])
            ws.write(i + 1, 11, 1 + averageGainSorted[i])
            ws.write(i + 1, 12, 1 + expectedGainSorted[i])
            ws.write(i + 1, 13, (6.0 + gainSorted[i] + fScoreSorted[i] + fScoreWeightedSorted[i] + averagefScoreWeightedSorted[i] + averageGainSorted[i] + expectedGainSorted[i]) / 6.0)
            ws.write(i + 1, 14, fi.AverageTreeIndex)
            ws.write(i + 1, 15, fi.AverageTreeDepth)

    interactions = FeatureInteractions.interactions_with_leaf_stats()
    if interactions:
        if verbosity >= 1:
            print("Writing leaf statistics")

        ws = workbook.add_worksheet("Leaf Statistics")

        ws.set_row(0, 20, cf_first_row)
        ws.set_column(0, 0, max([len(i.Name) for i in interactions]) + 10, cf_first_column)
        ws.set_column(1, 4, 20)

        for col, name in enumerate([
            "Interaction", "Sum Leaf Values Left", "Sum Leaf Values Right", "Sum Leaf Covers Left", "Sum Leaf Covers Right"
        ]):
            ws.write(0, col, name)

        for i, fi in enumerate(interactions):
            ws.write(i + 1, 0, fi.Name)
            ws.write(i + 1, 1, fi.SumLeafValuesLeft)
            ws.write(i + 1, 2, fi.SumLeafValuesRight)
            ws.write(i + 1, 3, fi.SumLeafCoversLeft)
            ws.write(i + 1, 4, fi.SumLeafCoversRight)

    interactions = FeatureInteractions.interactions_of_depth(0)
    if interactions:
        if verbosity >= 1:
            print("Writing split value histograms")

        ws = workbook.add_worksheet("Split Value Histograms")

        ws.set_row(0, 20, cf_first_row)
        ws.set_column(0, 0, max([len(i.Name) for i in interactions]) + 10, cf_first_column)
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

            ws.merge_range(0, c1, 0, c2, fi.Name)
            ws.set_column(c1, c1, max(10, (len(fi.Name) + 4) / 2))
            ws.set_column(c2, c2, max(10, (len(fi.Name) + 4) / 2))

            for j, key in enumerate(sorted(fi.SplitValueHistogram.values.keys())):
                ws.write(j + 1, c1, key)
                ws.write(j + 1, c2, fi.SplitValueHistogram.values[key])

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
        '-o', dest='OutputXlsxFile', action='store', default='XgbFeatureInteractions.xlsx',
        help='Xlsx file to be written')

    arg_parser.add_argument(
        '-t', dest='MaxTrees', action='store', default='100', type=int,
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
    args.OutputXlsxFile = args.OutputXlsxFile.strip()

    verbosity = int(args.Verbosity)

    settings_print = '''
Settings:
=========
XGBModelFile (-m): {model}
OutputXlsxFile (-o): {output}
MaxInteractionDepth: {depth}
MaxDeepening (-g): {deepening}
MaxTrees (-t): {trees}
TopK (-k): {topk}
SortBy (-s): {sortby}
MaxHistograms (-H): {histograms}
'''.format(
        model=args.XGBModelFile,
        output=args.OutputXlsxFile,
        depth=args.MaxInteractionDepth,
        deepening=args.MaxDeepening,
        trees=args.MaxTrees,
        topk=args.TopK,
        sortby=args.SortBy,
        histograms=args.MaxHistograms
    )

    if verbosity >= 1:
        print(settings_print)

    FeatureScoreComparer(args.SortBy)

    xgbParser = XGBModelParser(verbosity)
    xgbModel = xgbParser.model_from_file(args.XGBModelFile, args.MaxTrees)
    featureInteractions = xgbModel.GetFeatureInteractions(args.MaxInteractionDepth, args.MaxDeepening)

    FeatureInteractionsWriter(featureInteractions, args.OutputXlsxFile, args.MaxInteractionDepth, args.TopK, args.MaxHistograms)

    if verbosity >= 1:
        print(epilog)

    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


def saveXgbFI(booster, feature_names=None, OutputXlsxFile='XgbFeatureInteractions.xlsx', MaxTrees=100, MaxInteractionDepth=2, MaxDeepening=-1, TopK=100, MaxHistograms=10, SortBy='Gain'):
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
    FeatureScoreComparer(SortBy)
    xgbParser = XGBModelParser()
    dump = booster.get_dump('', with_stats=True)
    xgbModel = xgbParser.model_from_memory(dump, MaxTrees)
    featureInteractions = xgbModel.GetFeatureInteractions(MaxInteractionDepth, MaxDeepening)
    FeatureInteractionsWriter(featureInteractions, OutputXlsxFile, MaxInteractionDepth, TopK, MaxHistograms)


if __name__ == '__main__':
    entry_point()
