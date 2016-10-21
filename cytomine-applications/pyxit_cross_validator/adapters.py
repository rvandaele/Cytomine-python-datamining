# -*- coding: utf-8 -*-
import numpy as np
import urllib
from cytomine import Cytomine
from cytomine.models import AnnotationCollection
from pyxit.estimator import PyxitClassifier, _get_output_from_directory
from sklearn.svm import LinearSVC

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class AnnotationCollectionAdapter(object):
    def __init__(self, collection):
        if isinstance(collection, AnnotationCollection):
            self._annotations = collection.data()
        else:
            self._annotations = collection

    def data(self):
        return self._annotations

    def __len__(self):
        return len(self._annotations)

    def __iter__(self):
        for a in self._annotations:
            yield a

    def __iadd__(self, other):
        if isinstance(other, AnnotationCollection):
            self._annotations += other.data()
        return self


class PyxitClassifierAdapter(PyxitClassifier):

    def __init__(self, base_estimator,
                 n_subwindows=10,
                 min_size=0.5,
                 max_size=1.0,
                 target_width=16,
                 target_height=16,
                 n_jobs=1,
                 interpolation=2,
                 transpose=False,
                 colorspace=2,
                 fixed_size=False,
                 random_state=None,
                 verbose=0,
                 get_output=_get_output_from_directory,
                 window_sizes=None, **other_params):
        PyxitClassifier.__init__(self, base_estimator, n_subwindows=n_subwindows, min_size=min_size, max_size=max_size,
                                 target_width=target_width, target_height=target_height, n_jobs=n_jobs,
                                 interpolation=interpolation, transpose=transpose, colorspace=colorspace,
                                 fixed_size=fixed_size, random_state=random_state, verbose=verbose,
                                 get_output=get_output, **other_params)
        if window_sizes is not None:
            self.min_size = window_sizes[0]
            self.max_size = window_sizes[1]

    def get_params(self, deep=True):
        parameters = self.base_estimator.get_params() if deep else dict()
        parameters["base_estimator"] = self.base_estimator
        parameters["n_subwindows"] = self.n_subwindows
        parameters["min_size"] = self.min_size
        parameters["max_size"] = self.max_size
        parameters["window_sizes"] = self.min_size, self.max_size
        parameters["target_width"] = self.target_width
        parameters["target_height"] = self.target_height
        parameters["n_jobs"] = self.n_jobs
        parameters["interpolation"] = self.interpolation
        parameters["transpose"] = self.transpose
        parameters["colorspace"] = self.colorspace
        parameters["fixed_size"] = self.fixed_size
        parameters["verbose"] = self.verbose
        parameters["get_output"] = self.get_output
        return parameters

    def set_params(self, **params):
        super(PyxitClassifierAdapter, self).set_params(**dict(params))
        if "window_sizes" in params and params["window_sizes"] is not None:
            self.min_size, self.max_size = params["window_sizes"]
        return self

    def equivalent_pyxit(self):
        pyxit = PyxitClassifier(self.base_estimator, n_subwindows=self.n_subwindows, min_size=self.min_size,
                                max_size=self.max_size, target_width=self.target_width, target_height=self.target_height,
                                n_jobs=self.n_jobs, interpolation=self.interpolation, transpose=self.transpose,
                                colorspace=self.colorspace, fixed_size=self.fixed_size, random_state=self.random_state,
                                verbose=self.verbose, get_output=self.get_output)
        pyxit.n_classes_ = self.n_classes_
        pyxit.classes_ = self.classes_
        pyxit.maxs = self.maxs
        return pyxit


class SVMPyxitClassifierAdapter(PyxitClassifierAdapter):
    def __init__(self, base_estimator,
                 C=1.0,
                 n_subwindows=10,
                 min_size=0.5,
                 max_size=1.0,
                 target_width=16,
                 target_height=16,
                 n_jobs=1,
                 interpolation=2,
                 transpose=False,
                 colorspace=2,
                 fixed_size=False,
                 random_state=None,
                 verbose=0,
                 get_output=_get_output_from_directory,
                 window_sizes=None,
                 **other_params):
        PyxitClassifierAdapter.__init__(self, base_estimator, n_subwindows=n_subwindows, min_size=min_size,
                                        max_size=max_size, target_width=target_width, target_height=target_height,
                                        n_jobs=n_jobs, interpolation=interpolation, transpose=transpose,
                                        colorspace=colorspace, fixed_size=fixed_size, random_state=random_state,
                                        verbose=verbose, get_output=get_output, **other_params)
        if window_sizes is not None:
            self.min_size = window_sizes[0]
            self.max_size = window_sizes[1]
        self._svm = LinearSVC(C=C)

    def fit(self, X, y, _X=None, _y=None):
        if _X is None or _y is None:
            _X, _y = self.extract_subwindows(X, y)
        super(SVMPyxitClassifierAdapter, self).fit(X, y, _X=_X, _y=_y)
        Xt = self.transform(X, _X=_X, reset=True)
        self._svm.fit(Xt, y)

    def predict(self, X, _X=None):
        if _X is None:
            y = np.zeros(X.shape[0])
            _X, _ = self.extract_subwindows(X, y)
        Xt = self.transform(X, _X=_X)
        return self._svm.predict(Xt)

    def get_params(self, deep=True):
        params = super(SVMPyxitClassifierAdapter, self).get_params(deep=deep)
        params["C"] = self._svm.C
        return params

    def set_params(self, **params):
        super(PyxitClassifierAdapter, self).set_params(**dict(params))
        if "C" in params:
            self._svm.C = params["C"]
        return self

    @property
    def svm(self):
        return self._svm


class CytomineAdapter(Cytomine):
    def get_annotations(self, id_project=None, id_user=None, id_image=None, id_term=None, showGIS=None,
                        showWKT=None, showMeta=None, bbox=None, id_bbox= None, reviewed_only=False):
        if not reviewed_only:
            return super(CytomineAdapter, self).get_annotations(id_project=id_project, id_user=id_user,
                                                                id_image=id_image, id_term=id_term, showGIS=showGIS,
                                                                showWKT=showWKT, showMeta=showMeta, bbox=bbox,
                                                                id_bbox=id_bbox, reviewed_only=False)
        annotations = AnnotationCollection()
        query = ""
        if id_project:
            query += "&project=" + str(id_project)
        if id_term:
            query += "&terms=" + str(id_term)  # terms for multiple, term instead
        if id_user:
            query = "{}&{}={}".format(query, "reviewUsers" if reviewed_only else "users",
                                      str(id_user).strip('[]').replace(' ', ''))
        if id_image:
            query += "&images=%s" % str(id_image).strip('[]').replace(' ', '')
        if bbox:
            query += "&bbox=%s" % urllib.quote_plus(bbox)  # %str(bbox).strip('[]').replace(' ','%20')
        if id_bbox:
            query += "&bboxAnnotation=%s" % str(id_bbox)
        if showGIS:
            query += "&showGIS=true"
        if showWKT:
            query += "&showWKT=true"
        if showMeta:
            query += "&showMeta=true"
        if showMeta or showGIS or showWKT:
            query += "&showTerm=true"
        if reviewed_only:
            query += "&reviewed=true"

        # print query
        annotations = self.fetch(annotations, query=query)
        return annotations

if __name__ == "__main__":
    conn = CytomineAdapter("beta.cytomine.be", "ad014190-2fba-45de-a09f-8665f803ee0b",
                           "767512dd-e66f-4d3c-bb46-306fa413a5eb", base_path='/api/')
    data = conn.get_annotations(id_project=716498, id_user=14, reviewed_only=True, id_image=[8120444])
    print data.data()
    print len(data.data())
    print data.data()