"""Utilities to extract features from images"""

# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          James Bergstra <james.bergstra@umontreal.ca>
# License: BSD

import numpy as np
import math
from scipy import sparse
from ..utils.fixes import in1d
from ..base import BaseEstimator
from ..pca import PCA
from ..cluster import KMeans
from ..metrics.pairwise import euclidean_distances

################################################################################
# From an image to a graph

def _make_edges_3d(n_x, n_y, n_z=1):
    """Returns a list of edges for a 3D image.

    Parameters
    ===========
    n_x: integer
        The size of the grid in the x direction.
    n_y: integer
        The size of the grid in the y direction.
    n_z: integer, optional
        The size of the grid in the z direction, defaults to 1
    """
    vertices = np.arange(n_x*n_y*n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges


def _compute_gradient_3d(edges, img):
    n_x, n_y, n_z = img.shape
    gradient = np.abs(img[edges[0]/(n_y*n_z), \
                                (edges[0] % (n_y*n_z))/n_z, \
                                (edges[0] % (n_y*n_z))%n_z] - \
                           img[edges[1]/(n_y*n_z), \
                                (edges[1] % (n_y*n_z))/n_z, \
                                (edges[1] % (n_y*n_z)) % n_z])
    return gradient


# XXX: Why mask the image after computing the weights?

def _mask_edges_weights(mask, edges, weights):
    """Apply a mask to weighted edges"""
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(in1d(edges[0], inds),
                              in1d(edges[1], inds))
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    maxval = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(maxval+1))
    edges = order[edges]
    return edges, weights


def img_to_graph(img, mask=None, return_as=sparse.coo_matrix, dtype=None):
    """Graph of the pixel-to-pixel gradient connections

    Edges are weighted with the gradient values.

    Parameters
    ==========
    img: ndarray, 2D or 3D
        2D or 3D image
    mask : ndarray of booleans, optional
        An optional mask of the image, to consider only part of the
        pixels.
    return_as: np.ndarray or a sparse matrix class, optional
        The class to use to build the returned adjacency matrix.
    dtype: None or dtype, optional
        The data of the returned sparse matrix. By default it is the
        dtype of img
    """
    img = np.atleast_3d(img)
    if dtype is None:
        dtype = img.dtype
    n_x, n_y, n_z = img.shape
    edges   = _make_edges_3d(n_x, n_y, n_z)
    weights = _compute_gradient_3d(edges, img)
    if mask is not None:
        edges, weights = _mask_edges_weights(mask, edges, weights)
        img = img.squeeze()[mask]
    else:
        img = img.ravel()
    n_voxels = img.size
    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))
    graph = sparse.coo_matrix((np.hstack((weights, weights, img)),
                              (np.hstack((i_idx, diag_idx)),
                               np.hstack((j_idx, diag_idx)))),
                              (n_voxels, n_voxels),
                              dtype=dtype)
    if return_as is np.ndarray:
        return graph.todense()
    return return_as(graph)


################################################################################
# From an image to a set of small image patches

def extract_patches_2d(images, image_size, patch_size, max_patches=None):
    """Reshape a collection of 2D images into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Parameters
    ----------
    images: array with shape (n_images, i_h, i_w) or (n_images, i_h * i_w)
        the original image data

    image_size: tuple of ints (i_h, i_w)
        the dimensions of the images

    patch_size: tuple of ints (p_h, p_w)
        the dimensions of one patch

    max_patches: integer, optional default is None
        the maximum number of patches to extract

    Returns
    -------
    patches: array
         shape is (n_patches, patch_height, patch_width, n_colors)
         or (n_patches, patch_height, patch_width) if n_colors is 1

    Examples
    --------

    >>> one_image = np.arange(16).reshape((1, 4, 4))
    >>> one_image
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]]])

    >>> patches = extract_patches_2d(one_image, (4, 4), (2, 2))
    >>> patches.shape
    (9, 2, 2)

    >>> patches[0]
    array([[0, 1],
           [4, 5]])

    >>> patches[1]
    array([[1, 2],
           [5, 6]])

    >>> patches[8]
    array([[10, 11],
           [14, 15]])

    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patch_size

    images = np.atleast_2d(images)
    # ensure the images have the usual shape, including explicit color channel
    n_images = images.shape[0]
    images = images.reshape((n_images, i_h, i_w, -1))
    n_colors = images.shape[-1]

    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    n_patches_by_image = n_h * n_w

    # optional bound on the memory allocation
    if max_patches and n_patches_by_image * n_images > max_patches:
        if n_patches_by_image > max_patches:
            raise ValueError("unsatisfailable constraint: "
                             "n_patches_by_image=%d and "
                             "max_patches=%d" % (
                                 n_patches_by_image, max_patches))
        # only use a subset of the images to extract patches from
        n_images = int(max_patches / n_patches_by_image)
        images = images[:n_images]

    # effective number of patches
    n_patches = n_images * n_patches_by_image

    # extract the patches
    patches = np.zeros((n_patches, p_h, p_w, n_colors), dtype=images.dtype)
    offset = 0
    for i in xrange(n_h):
        for j in xrange(n_w):
            start = offset * n_images
            stop = start + n_images
            patches[start:stop, :, :, :] = images[:, i:i + p_h, j:j + p_w, :]
            offset +=1

    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches


class ConvolutionalKMeansEncoder(BaseEstimator):
    """Unsupervised sparse feature extractor for 2D images

    The fit method extracts patches from the images, whiten them using
    a PCA transform and run a KMeans algorithm to extract the patch
    cluster centers.

    The input is then correlated with each individual "patch-center"
    treated as a convolution kernel. The activations are then
    sparse-encoded using the "triangle" variant of a KMeans such that
    for each center c[k] we define a transformation function f_k over
    the input patches:

        f_k(x) = max(0, np.mean(z) - z[k])

    where z[k] = linalg.norm(x - c[k]) and x is an input patch.

    Activations are then sum-pooled over the 4 quadrants of the original
    image space.

    The transform operation is performed by applying the triangle kmeans
    sparse-encoding and sum-pooling.

    This estimator only implements the unsupervised feature extraction
    part of the referenced paper. Image classification can then be
    performed by training a linear SVM model on the output of this
    estimator.

    Parameters
    ----------
    n_centers: int, optional: default 400
        number of centers extracted by the kmeans algorithm

    patch_size: tuple of int, optional: default 6
        the size of the square patches / convolution kernels learned by
        kmeans

    step_size: int, optional: 1
        number of pixels to shift between two consecutive patches (a.k.a.
        stride)

    whiten: boolean, optional: default True
        perform a whitening PCA on the patches at feature extraction time

    n_pools: int, optional: default 2
        number equal size areas to perform the sum-pooling of features
        over: n_pools=2 means 4 quadrants, n_pools=3 means 6 areas and so on

    local_contrast: boolean, optional: default True
        perform local contrast normalization on the extracted patch

    Reference
    ---------
    An Analysis of Single-Layer Networks in Unsupervised Feature Learning
    Adam Coates, Honglak Lee and Andrew Ng. In NIPS*2010 Workshop on
    Deep Learning and Unsupervised Feature Learning.
    http://robotics.stanford.edu/~ang/papers/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf
    """

    def __init__(self, n_centers=400, image_size=None, patch_size=6,
                 step_size=1, whiten=True, n_components=None, max_patches=1e5,
                 n_pools=2, max_iter=100, n_init=1, n_prefit=5, tol=1e-1,
                 local_contrast=True, n_drop_components=0, verbose=False):
        self.n_centers = n_centers
        self.patch_size = patch_size
        self.step_size = step_size
        self.whiten = whiten
        self.n_pools = n_pools
        self.image_size = image_size
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_components = n_components
        self.max_patches = int(max_patches)
        self.local_contrast = local_contrast
        self.n_prefit = n_prefit
        self.verbose = verbose
        self.tol = tol
        self.n_drop_components = n_drop_components

    def _check_images(self, X):
        """Check that X can seen as a consistent collection of images"""
        X = np.atleast_2d(X)
        n_samples = X.shape[0]

        if self.image_size is None:
            if len(X.shape) == 4:
                self.image_size = X.shape[1:3]
                self.n_colors = X.shape[3]
            elif len(X.shape) == 3:
                self.image_size = X.shape[1:3]
                self.n_colors = 1
            elif len(X.shape) == 2:
                # assume square images in gray levels
                _, n_features = X.shape
                size = int(math.sqrt(n_features))
                if size ** 2 != n_features:
                    raise ValueError("images with shape %r are not squares: "
                                     "the image size must be made explicit" %
                                     (X.shape,))
                self.image_size = (size, size)
                self.n_colors = 1
            else:
                raise ValueError("image set should have shape maximum 4d")

        return X.reshape(
            (n_samples, self.image_size[0], self.image_size[1], self.n_colors))

    def local_contrast_normalization(self, patches):
        """Normalize the patch-wise variance of the signal"""
        # center all colour channels together
        patches = patches.reshape((patches.shape[0], -1))
        patches -= patches.mean(axis=1)[:, None]

        patches_std = patches.std(axis=1)
        # Cap the divisor to avoid amplifying patches that are essentially
        # a flat surface into full-contrast salt-and-pepper garbage.
        # the actual value is a wild guess
        # This trick is credited to N. Pinto
        min_divisor = (2 * patches_std.min() + patches_std.mean()) / 3
        patches /= np.maximum(min_divisor, patches_std).reshape(
            (patches.shape[0], 1))
        return patches

    def fit(self, X):
        """Fit the feature extractor on a collection of 2D images"""
        X = self._check_images(X)
        patch_size = (self.patch_size, self.patch_size)

        # extract the patches who will be clustered into filters
        if self.verbose:
            print "Extracting patches from images"
        patches = extract_patches_2d(X, self.image_size, patch_size,
                                     max_patches=self.max_patches)
        n_patches = patches.shape[0]
        patches = patches.reshape((n_patches, -1))


        # normalize each patch individually
        if self.local_contrast:
            if self.verbose:
                print "Local contrast normalization of the patches"
            patches = self.local_contrast_normalization(patches)

        # kmeans model to find the filters
        if self.verbose:
            print "About to extract filters from %d patches" % n_patches
        kmeans = KMeans(k=self.n_centers, init='k-means++',
                        max_iter=self.max_iter, n_init=self.n_init,
                        tol=self.tol, verbose=self.verbose)

        if self.whiten:
            # whiten the patch space
            if self.verbose:
                print "Whitening PCA of the patches"
            self.pca = pca = PCA(whiten=True, n_components=self.n_components)
            pca.fit(patches)

            # implement a band-pass filter by dropping the first eigen
            # values which are generally low frequency components
            drop = self.n_drop_components
            if drop:
                pca.components_ = pca.components_[:, drop:]
            patches = pca.transform(patches)

            # compute the KMeans centers
            if 0 < self.n_prefit < patches.shape[1]:
                if self.verbose:
                    print "First KMeans in simplified curriculum space"
                # starting the kmeans on a the projection to the first singular
                # components: curriculum learning trick by Andrej Karpathy
                kmeans.fit(patches[:, :self.n_prefit])

                # warm restart by padding previous centroids with zeros
                # with full dimensionality this time
                kmeans.init = np.zeros((self.n_centers, patches.shape[1]),
                                       dtype=kmeans.cluster_centers_.dtype)
                kmeans.init[:, :self.n_prefit] = kmeans.cluster_centers_
                if self.verbose:
                    print "Second KMeans in full whitened patch space"
                kmeans.fit(patches, n_init=1)
            else:
                if self.verbose:
                    print "KMeans in full original patch space"
                # regular kmeans fit (without the curriculum trick)
                kmeans.fit(patches)

            # project back the centers in original, non-whitened space (useful
            # for qualitative inspection of the filters)
            self.filters_ = self.pca.inverse_transform(kmeans.cluster_centers_)
        else:
            # find the kernel in the raw original dimensional space
            # TODO: experiment with component wise scaling too
            self.pca = None
            kmeans.fit(patches)
            self.filters_ = kmeans.cluster_centers_

        self.kmeans = kmeans
        self.inertia_ = kmeans.inertia_
        return self

    def transform(self, X):
        """Map a collection of 2D images into the feature space"""
        X = self._check_images(X)
        n_samples, n_rows, n_cols, n_channels = X.shape
        n_filters = self.filters_.shape[0]
        ps = self.patch_size

        pooled_features = np.zeros((X.shape[0], self.n_pools, self.n_pools,
                                    n_filters), dtype=X.dtype)

        n_rows_adjusted = n_rows - self.patch_size + 1
        n_cols_adjusted = n_cols - self.patch_size + 1

        for r in xrange(n_rows_adjusted):
            if self.verbose:
                print "Extracting features for row #%d/%d" % (
                    r + 1, n_rows_adjusted)

            for c in xrange(n_cols_adjusted):

                patches = X[:, r:r + ps, c:c + ps, :].reshape((n_samples, -1))

                if self.local_contrast:
                    # TODO: make it inplace by default explictly
                    patches = self.local_contrast_normalization(patches)

                if self.whiten:
                    # TODO: make it possible to pass pre-allocated array
                    patches = self.pca.transform(patches)

                # extract distance from each patch to each cluster center
                # TODO: make it possible to reuse pre-allocated distance array
                filters = self.kmeans.cluster_centers_
                distances = euclidean_distances(patches, filters)

                # triangle features
                distance_means = distances.mean(axis=1)[:, None]
                features = np.maximum(0, distance_means - distances)

                # features are pooled over image regions
                out_r = r * self.n_pools / n_rows_adjusted
                out_c = c * self.n_pools / n_cols_adjusted
                pooled_features[:, out_r, out_c, :] += features

        # downstream classifiers expect a 2 dim shape
        return pooled_features.reshape(pooled_features.shape[0], -1)

