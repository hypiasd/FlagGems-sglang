"""Common output and input invariants, independent of operator semantics."""
def check_output(got, expected, atol=0, rtol=0):
    import torch
    if isinstance(expected, (tuple, list)):
        if not isinstance(got, type(expected)) or len(got) != len(expected):
            raise AssertionError("wrong output structure")
        for actual, want in zip(got, expected):
            check_output(actual, want, atol, rtol)
        return
    if not isinstance(got, torch.Tensor) or got.shape != expected.shape or got.dtype != expected.dtype or got.device != expected.device:
        raise AssertionError("wrong output shape, dtype or device")
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol, equal_nan=True)


def snapshot_inputs(inputs):
    import torch
    return [(tensor, tuple(tensor.shape), tensor.stride(), tensor.storage_offset(),
             tensor._version, tensor.new_empty(0).set_(tensor.storage()).view(torch.uint8).clone())
            for tensor in inputs if isinstance(tensor, torch.Tensor)]


def check_inputs(snapshots):
    import torch
    for tensor, shape, stride, offset, version, before in snapshots:
        if tuple(tensor.shape) != shape or tensor.stride() != stride or tensor.storage_offset() != offset or tensor._version != version:
            raise AssertionError("input metadata/version changed")
        after = tensor.new_empty(0).set_(tensor.storage()).view(torch.uint8)
        if not torch.equal(before, after):
            raise AssertionError("input backing storage was mutated")


def mirror(cpu, device):
    backing = cpu.new_empty(0).set_(cpu.storage()).to(device, copy=True)
    return backing.as_strided(cpu.shape, cpu.stride(), cpu.storage_offset())
