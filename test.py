import numpy as np
import torch
import pytest
from engine import Value, match_shape

torch.set_default_dtype(torch.float64)
np.random.seed(42)

# Helpers

def to_torch(arr):
    """Convert numpy array or scalar to torch tensor with grad."""
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)

def assert_allclose(a, b, *, msg=""):
    assert np.allclose(a, b, rtol=1e-6, atol=1e-6), msg or f"\n{a}\nvs\n{b}"

def compare_leaf_grads(val_leaves, torch_leaves):
    """Utility: make sure all leaves have matching grads."""
    for v, t in zip(val_leaves, torch_leaves):
        assert_allclose(v.grad, t.grad.detach().numpy(),
                        msg=f"Gradient mismatch for leaf with data {v.data}")


# Fixtures that generate random test inputs of varying shapes

@pytest.fixture(params=[
    (),                   # scalar
    (7,),                 # vector
    (3, 5),               # matrix
    (2, 4, 3)             # 3-D tensor
])
def rand_shape(request):
    return request.param

def rand_np(shape):
    return np.random.randn(*shape)


# Elementary operations

def test_add(rand_shape):
    a_np, b_np = rand_np(rand_shape), rand_np(rand_shape)
    a_val, b_val = Value(a_np), Value(b_np)
    a_tch, b_tch = to_torch(a_np), to_torch(b_np)

    out_val = a_val + b_val
    out_tch = a_tch + b_tch

    out_val.backward()
    out_tch.backward(torch.ones_like(out_tch))

    # data
    assert_allclose(out_val.data, out_tch.detach().numpy())
    # grads
    compare_leaf_grads([a_val, b_val], [a_tch, b_tch])

def test_add_broadcast():
    a_np = rand_np((2, 3, 4))
    b_np = rand_np((4,))
    a_val, b_val = Value(a_np), Value(b_np)
    a_tch, b_tch = to_torch(a_np), to_torch(b_np)

    out_val = a_val + b_val
    out_tch = a_tch + b_tch
    out_val.sum().backward()
    out_tch.sum().backward()

    compare_leaf_grads([a_val, b_val], [a_tch, b_tch])

def test_mul(rand_shape):
    a_np, b_np = rand_np(rand_shape), rand_np(rand_shape)
    a_val, b_val = Value(a_np), Value(b_np)
    a_tch, b_tch = to_torch(a_np), to_torch(b_np)

    out_val = a_val * b_val
    out_tch = a_tch * b_tch
    out_val.backward()
    out_tch.backward(torch.ones_like(out_tch))

    compare_leaf_grads([a_val, b_val], [a_tch, b_tch])

def test_div(rand_shape):
    a_np, b_np = rand_np(rand_shape), rand_np(rand_shape) + 1.1
    a_val, b_val = Value(a_np), Value(b_np)
    a_tch, b_tch = to_torch(a_np), to_torch(b_np)

    out_val = a_val / b_val
    out_tch = a_tch / b_tch
    out_val.backward()
    out_tch.backward(torch.ones_like(out_tch))

    compare_leaf_grads([a_val, b_val], [a_tch, b_tch])

def test_pow_scalar(rand_shape):
    a_np = rand_np(rand_shape)
    exp = 3.0
    a_val, a_tch = Value(a_np), to_torch(a_np)

    out_val = a_val ** exp
    out_tch = a_tch ** exp
    out_val.backward()
    out_tch.backward(torch.ones_like(out_tch))

    compare_leaf_grads([a_val], [a_tch])

def test_neg_sub(rand_shape):
    x_np, y_np = rand_np(rand_shape), rand_np(rand_shape)
    x_val, y_val = Value(x_np), Value(y_np)
    x_tch, y_tch = to_torch(x_np), to_torch(y_np)

    out_val = x_val - y_val
    out_tch = x_tch - y_tch
    out_val.backward()
    out_tch.backward(torch.ones_like(out_tch))
    compare_leaf_grads([x_val, y_val], [x_tch, y_tch])


# Dot and MatMul

def test_dot():
    x_np, y_np = rand_np((5,)), rand_np((5,))
    x_val, y_val = Value(x_np), Value(y_np)
    x_tch, y_tch = to_torch(x_np), to_torch(y_np)

    out_val = x_val.dot(y_val)
    out_tch = torch.dot(x_tch, y_tch)
    out_val.backward()
    out_tch.backward()
    compare_leaf_grads([x_val, y_val], [x_tch, y_tch])

@pytest.mark.parametrize("m,n,p", [(3,4,2), (2,5,7)])
def test_matmul(m, n, p):
    a_np = rand_np((m, n))
    b_np = rand_np((n, p))
    a_val, b_val = Value(a_np), Value(b_np)
    a_tch, b_tch = to_torch(a_np), to_torch(b_np)

    out_val = a_val @ b_val
    out_tch = a_tch @ b_tch
    loss_val = out_val.sum()
    loss_tch = out_tch.sum()
    loss_val.backward()
    loss_tch.backward()
    compare_leaf_grads([a_val, b_val], [a_tch, b_tch])


# Reduction and activation

def test_sum_keepdims():
    x_np = rand_np((2,3,4))
    x_val, x_tch = Value(x_np), to_torch(x_np)

    out_val = x_val.sum(axis=1, keepdims=True)
    out_tch = x_tch.sum(dim=1, keepdim=True)
    out_val.backward()
    out_tch.backward(torch.ones_like(out_tch))
    compare_leaf_grads([x_val], [x_tch])

def test_relu():
    x_np = rand_np((6,7)) - 0.5
    x_val, x_tch = Value(x_np), to_torch(x_np)

    out_val = x_val.relu()
    out_tch = torch.relu(x_tch)
    out_val.backward()
    out_tch.backward(torch.ones_like(out_tch))
    compare_leaf_grads([x_val], [x_tch])


# Complex composite expression & zero_grad

def test_complex_and_zero_grad():
    a_np, b_np, c_np = rand_np((4,)), rand_np((4,)), rand_np((4,))
    a_val, b_val, c_val = Value(a_np), Value(b_np), Value(c_np)
    a_tch, b_tch, c_tch = to_torch(a_np), to_torch(b_np), to_torch(c_np)

    expr_val = ((a_val * b_val + c_val).dot(b_val) / 3).relu()
    expr_tch = ((a_tch * b_tch + c_tch).dot(b_tch) / 3).relu()

    expr_val.backward()
    expr_tch.backward()

    compare_leaf_grads([a_val, b_val, c_val], [a_tch, b_tch, c_tch])

    # check zero_grad
    for v in (expr_val, a_val, b_val, c_val):
        v.zero_grad()
        assert np.all(v.grad == 0), "zero_grad failed"


# match_shape utility – quick sanity check

@pytest.mark.parametrize("gshape,tshape", [
    ((2,3,4), (4,)),
    ((3,1,5), (3,1,5)),
    ((5,4),   (1,4)),
])
def test_match_shape(gshape, tshape):
    g = np.random.randn(*gshape)
    tgt = match_shape(g, tshape)
    assert tgt.shape == tshape