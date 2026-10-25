from retention_agent.schemas import CustomerRecord


def test_data_shapes(world):
    ds = world["ds"]
    assert len(ds.customers) == 500
    assert len(ds.offers) == 15 and len(ds.rules) >= 20 and len(ds.transcripts) == 60
    assert len(ds.golden) >= 20
    for row in ds.customer_rows()[:50]:
        CustomerRecord(**row)  # every generated row satisfies the contract


def test_labels_hidden_from_customer_features(world):
    assert "latent_reason" not in world["ds"].customers.columns
    assert "churned_30d" not in world["ds"].customers.columns


def test_index_counts(world):
    counts = world["store"].counts()
    assert counts == {"offers": 15, "transcripts": 120, "policies": 33}


def test_reason_filter_and_semantic_hits(world):
    store = world["store"]
    hits = store.query("transcripts", "dropped calls and slow data at home", 5, where={"reason": "network_quality"})
    assert hits and all(h.meta["reason"] == "network_quality" for h in hits)
    offers = store.query("offers", "network problems dropped calls credit", 3)
    assert any("network" in h.meta["targets"] for h in offers)


def test_generation_is_deterministic(settings, tmp_path):
    from dataclasses import replace
    from retention_agent.data_gen import build_customers, build_offers
    a, _, _ = build_customers(42, 500, build_offers())
    b, _, _ = build_customers(42, 500, build_offers())
    assert a.equals(b)
