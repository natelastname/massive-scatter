from massive_scatter.source import synthetic_batches


def test_synthetic_source_is_streamed_and_square():
    batches = list(synthetic_batches(100, batch_size=17, origin_x=10, origin_y=20))
    assert sum(batch.num_rows for batch in batches) == 100
    assert max(value.as_py() for batch in batches for value in batch.column(0)) == 109
    ys = [value.as_py() for batch in batches for value in batch.column(1)]
    assert min(ys) == 20
    assert max(ys) == 119
