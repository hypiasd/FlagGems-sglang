"""Address binding in the CPU semantic model.

A candidate may hand Triton raw integer addresses instead of tensors (the
``tl.cast(addr, tl.pointer_type(...))`` idiom).  Measured on the T4 that is worth
0.60 us per launch, because the launch path no longer holds a reference to the
output tensor the call just allocated -- it is the only rewrite of this wrapper
that ever produced a same-session win.  The model must therefore keep applying
every pointer check to those launches: each address is mapped back to the
allocation it names, and an integer that names nothing is left alone rather than
silently accepted as a pointer.
"""

import unittest

import torch

from competition.experiments import cpu_model


class AddressBinding(unittest.TestCase):
    def test_a_raw_address_is_rebound_to_the_tensor_it_names(self):
        tensor = torch.zeros(4)
        call = cpu_model.ValidationCall([tensor])
        self.assertIs(call.tensor_for_address(tensor.data_ptr()), tensor)

    def test_an_integer_that_names_nothing_is_not_a_pointer(self):
        # Constexpr integers (ROW_BLOCK, a dtype code) travel through the same
        # binding loop; they must not be mistaken for addresses, and an arbitrary
        # address must not become a pointer just because it is an int.
        call = cpu_model.ValidationCall([torch.zeros(4)])
        self.assertIsNone(call.tensor_for_address(1024))
        self.assertIsNone(call.tensor_for_address(0))
        self.assertIsNone(call.tensor_for_address(0xDEADBEEF))

    def test_the_output_allocation_is_reachable_by_address(self):
        # The member allocates the output inside the call and passes its address,
        # so register_output must publish that address; otherwise the "written
        # exactly once" check would never run on this candidate at all.
        tensor = torch.empty(6)
        call = cpu_model.ValidationCall([torch.zeros(2)])
        call.register_output(tensor)
        self.assertIs(call.tensor_for_address(tensor.data_ptr()), tensor)
        self.assertFalse(call.allocations[cpu_model.storage_key(tensor)].readonly)

    def test_a_view_address_maps_to_the_view_not_the_base(self):
        base = torch.zeros(8)
        view = base[2:6]
        call = cpu_model.ValidationCall([base, view])
        self.assertIs(call.tensor_for_address(view.data_ptr()), view)
        self.assertIs(call.tensor_for_address(base.data_ptr()), base)
        self.assertNotEqual(view.data_ptr(), base.data_ptr())

    def test_the_pointer_cast_is_the_identity_under_the_model(self):
        # ``tl.pointer_type`` only ever appears as ``tl.cast``'s second argument,
        # and the binding loop has already produced the Ptr, so the cast must not
        # rewrite it.
        model = cpu_model.CPUModel()
        tensor = torch.zeros(4)
        call = cpu_model.ValidationCall([tensor])
        pointer = call.pointer(tensor)
        self.assertIs(model.tl.cast(pointer, model.tl.pointer_type(model.tl.float32)), pointer)


if __name__ == "__main__":
    unittest.main()
