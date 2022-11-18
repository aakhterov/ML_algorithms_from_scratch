import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor 
from multiprocessing import Pool, cpu_count


def get_entropy(y):
    """
    Evaluate entropy for vector
    """
    _, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    fractions = counts/total
    return abs(-np.sum(fractions*np.log2(fractions)))


def get_gini_index_by_feature_value(y):
    _, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    fractions = counts/total
    return 1-np.sum(fractions**2)


def evaluate_gini_index_for_threhold(data):
    X_feature, y, thr = data
    y_by_feature_value = y[X_feature <= thr]
    w1 = len(y_by_feature_value)/len(y)
    less_or_equal = get_gini_index_by_feature_value(y_by_feature_value)
    y_by_feature_value = y[X_feature > thr]
    w2 = len(y_by_feature_value)/len(y)
    greater = get_gini_index_by_feature_value(y_by_feature_value)
    return w1*less_or_equal + w2*greater   


class Node:
    """
    Сlass for storing tree node data
    """

    def __init__(self, cost, class_, classes=None, classes_distrib=None, childs=None, depth=0):
        self.cost = cost
        self.class_ = class_
        self.split_feature = None
        self.threshold = None
        self.parent_feature_value = None
        self.classes = [] if classes is None else classes
        self.classes_distrib = [] if classes_distrib is None else classes_distrib            
        self.childs = [] if childs is None else childs
        self.depth=depth
    
    def __repr__(self):
        mn = self.depth*3
        return  f"{(mn*' ')}Value of parent's feature: {self.parent_feature_value}\n"                f"{(mn*' ')}Node class: {self.class_}\n"                f"{(mn*' ')}Node cost: {self.cost}\n"                f"{(mn*' ')}Classes in node: {self.classes}\n"                f"{(mn*' ')}Classes distribution in node: {self.classes_distrib}\n"                f"{(mn*' ')}Is leaf: {('False' if len(self.childs) else 'True')}\n"                f"{(mn*' ')}Feature selected fo split: {self.split_feature}\n"                f"{(mn*' ')}Threshold splitting: {self.threshold}\n"                f"{(mn*' ')}childs:\n{(mn*' ')}{self.childs}\n"
        
class DecisionTree:
    
    """
    Class for fitting Decision tree for classification tasks. The built tree is stored in nodes, which 
    are implemented using class Node. Features can be as categorical as continuous.  
    """
    
    def __init__(self, criterion="gini", max_depth=10, min_samples_for_splitting=2):
        if criterion =="entropy":
            self.__select_feature_function = self.__select_feature_with_max_inform_gain
            self.__evalute_criterion_function = get_entropy
        elif criterion =="gini":
            self.__select_feature_function = self.__select_feature_with_min_gini_index
            self.__evalute_criterion_function = get_gini_index_by_feature_value
        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")
            
        self.root = None
        self.max_depth = max_depth
        self.min_samples_for_splitting=min_samples_for_splitting
        
    def __get_thresholds(self, X_feature, y):
        thresholds = []
        X_feature_sorted = X_feature.sort_values()
        current_target_value = y.loc[X_feature_sorted.index[0]]
        for i, idx in enumerate(X_feature_sorted.index[1:]):
            if np.all(y.loc[idx] != current_target_value):
                current_target_value = y.loc[idx]
                if X_feature_sorted.loc[idx] != X_feature_sorted.loc[X_feature_sorted.index[i]]:
                    thresholds.append((X_feature_sorted.loc[idx] + X_feature_sorted.loc[X_feature_sorted.index[i]])/2)
        return sorted(list(set(thresholds)))

    def __get_gini_index(self, X_feature, y):      
        threshold = None
        if X_feature.dtypes == np.int64 or X_feature.dtypes == np.float64:
            gini_index = np.inf
            possible_thresholds = self.__get_thresholds(X_feature, y)
            if not possible_thresholds:
                return np.inf, None
            data = [(X_feature, y, thr) for thr in possible_thresholds]
            with Pool(cpu_count()-1) as pool:
                gs = pool.map(evaluate_gini_index_for_threhold, data)   
            gini_index = np.min(gs)
            threshold = possible_thresholds[np.argmin(gs)]        
        else:
            gini_index = 0
            unique_values, unique_counts = np.unique(X_feature, return_counts=True)
            weights = unique_counts/np.sum(unique_counts)
            for idx, value in enumerate(unique_values):
                y_by_feature_value = y[X_feature == value]
                gini_index +=weights[idx]*get_gini_index_by_feature_value(y_by_feature_value)
        return gini_index, threshold    
    
    def __get_information_gain(self, X_feature, y):
        entropy_before = get_entropy(y)
        entropy_after = 0
        unique_values, unique_counts = np.unique(X_feature, return_counts=True)
        weights = unique_counts/np.sum(unique_counts)
        for idx, value in enumerate(unique_values):
            y_by_feature = y[X_feature == value]
            entropy_after +=weights[idx]*get_entropy(y_by_feature)
        return entropy_before - entropy_after
    
    def __select_feature_with_max_inform_gain(self, X, y):
        igs = []
        for col in X.columns:
            ig = self.__get_information_gain(X[col], y)
            igs.append(ig)
        return X.columns[np.argmax(igs)]
    
    def __select_feature_with_min_gini_index(self, X, y):
        gis = []
        with ThreadPoolExecutor(max_workers=len(X.columns)) as executor:
            for col in X.columns:
                gis.append(executor.submit(self.__get_gini_index, X[col], y).result())            
        idx = np.argmin([el[0] for el in gis])
        return X.columns[idx], gis[idx][1]
    
    def __split_node(self, X, y, depth):
        classes, counts = np.unique(y, return_counts=True)
        cost = self.__evalute_criterion_function(y)
        node = Node(cost=cost, class_=classes[np.argmax(counts)], classes=classes, classes_distrib=counts, depth=depth) 
        if len(classes)==1 or depth>=self.max_depth or np.sum(counts)<self.min_samples_for_splitting:
            return node
        current_feature, threshold = self.__select_feature_function(X, y)
        node.split_feature = current_feature
        node.threshold = threshold
        if threshold is None:
            unique_values = np.unique(X[current_feature])
            for value in unique_values:
                X_after_split = X[X[current_feature]==value]
                y_after_split = y[X_after_split.index]
                child = self.__split_node(X_after_split, y_after_split, depth+1)
                child.parent_feature_value = value
                node.childs.append(child)
        else: 
            X_after_split_le = X[X[current_feature]<=threshold]
            y_after_split = y[X_after_split_le.index]
            child = self.__split_node(X_after_split_le, y_after_split, depth+1)
            child.parent_feature_value = None
            node.childs.append(child)
            X_after_split_g = X[X[current_feature]>threshold]
            y_after_split = y[X_after_split_g.index]
            child = self.__split_node(X_after_split_g, y_after_split, depth+1)
            child.parent_feature_value = None
            node.childs.append(child)
        return node
        
    def fit(self, X, y):
        self.root = self.__split_node(X, y, 0)      
    
    def __get_next_node(self, current_node, X):
        splitted_value = X[current_node.split_feature]
        if current_node.threshold is None:
            for child in current_node.childs:
                if child.parent_feature_value == splitted_value:
                    return child
        else:
            return current_node.childs[0] if splitted_value<=current_node.threshold else current_node.childs[1]
        return None
    
    def predict(self, X):
        y, ways = [], [] 
        for i in range(len(X)):
            path = []
            current_node = self.root
            is_leaf = False if len(current_node.childs) else True
            while not is_leaf:
                parent_node = current_node
                current_node = self.__get_next_node(current_node, X.iloc[i])
                if current_node is not None:
                    path.append({"feature": parent_node.split_feature, "value": current_node.parent_feature_value})
                is_leaf = False if (current_node is not None and len(current_node.childs)) else True
            y.append(current_node.class_ if current_node else parent_node.class_)
            ways.append(path)
        return y, ways
 