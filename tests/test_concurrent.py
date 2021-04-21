from hexrd.utils.concurrent import distribute_tasks


def test_distribute_tasks():
    num_tasks = 4
    max_workers = 4
    expected_result = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    result = distribute_tasks(num_tasks, max_workers)
    assert result == expected_result

    num_tasks = 3
    max_workers = 5
    expected_result = [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    result = distribute_tasks(num_tasks, max_workers)
    assert result == expected_result

    num_tasks = 5
    max_workers = 3
    expected_result = [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    result = distribute_tasks(num_tasks, max_workers)
    assert result == expected_result

    num_tasks = 574
    max_workers = 24
    expected_result = [
        (0, 24),
        (24, 48),
        (48, 72),
        (72, 96),
        (96, 120),
        (120, 144),
        (144, 168),
        (168, 192),
        (192, 216),
        (216, 240),
        (240, 264),
        (264, 288),
        (288, 312),
        (312, 336),
        (336, 360),
        (360, 384),
        (384, 408),
        (408, 432),
        (432, 456),
        (456, 480),
        (480, 504),
        (504, 528),
        (528, 551),
        (551, 574),
    ]
    result = distribute_tasks(num_tasks, max_workers)
    assert result == expected_result
