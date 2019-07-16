# Authors: Christian Lorentzen <lorentzen.ch@gmail.com>
#
# License: BSD 3 clause
import numpy as np
from numpy.testing import assert_allclose
import pytest

from sklearn.linear_model._glm.link import (
    IdentityLink,
    LogLink,
    LogitLink,
)


LINK_FUNCTIONS = [IdentityLink, LogLink, LogitLink]


@pytest.mark.parametrize('link', LINK_FUNCTIONS)
def test_link_properties(link):
    """Test link inverse and derivative."""
    rng = np.random.RandomState(42)
    x = rng.rand(100)*100
    link = link()  # instantiate object
    if isinstance(link, LogitLink):
        # careful for large x, note expit(36) = 1
        # limit max eta to 15
        x = x / 100 * 15
    assert_allclose(link.link(link.inverse(x)), x)
    # if f(g(x)) = x, then f'(g(x)) = 1/g'(x)
    assert_allclose(link.derivative(link.inverse(x)),
                    1./link.inverse_derivative(x))

    assert (
      link.inverse_derivative2(x).shape == link.inverse_derivative(x).shape)

    # for LogitLink, in the following x should be between 0 and 1.
    # assert_almost_equal(link.inverse_derivative(link.link(x)),
    #                     1./link.derivative(x), decimal=decimal)
