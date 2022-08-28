import torch


def test_torch_batch_matmul():
    a = torch.randint(10, (1, 2, 3, 4))
    b = torch.randint(10, (1, 2, 4, 4))

    def test1():
        res = a @ b
        return res

    def test2():
        return a.view(-1, 3, 4).bmm(b.view(-1, 4, 4)).reshape(1, 2, 3, 4)

    def test3():
        q = []
        for i in range(a.shape[0]):
            w = []
            for j in range(a.shape[1]):
                w.append(a[i, j] @ b[i, j])

            q.append(torch.stack(w))

        return torch.stack(q)

    assert test1().shape == test2().shape == test3().shape
    assert test1().equal(test2())
    assert test2().equal(test3())
