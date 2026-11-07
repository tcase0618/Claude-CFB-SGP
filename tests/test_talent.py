import numpy as np


def test_posterior_shrinkage(conn):
    from cfbquant.analytics.talent import posteriors
    p = posteriors(conn, 2025)
    assert not p.empty
    none = p[p["sample"].fillna(0) == 0]
    if len(none):
        assert np.allclose(none.post_mean, none.prior_mean)
        assert np.allclose(none.post_sd, none.prior_sd)
    heavy = p[p["sample"].fillna(0) > 150]
    if len(heavy):
        assert (heavy.post_sd < heavy.prior_sd).all()
        # heavy samples are pulled close to observed production
        assert np.corrcoef(heavy.post_mean, heavy.adj_epa_play)[0, 1] > 0.8
