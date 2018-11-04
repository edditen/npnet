import numpy as np
import neuralnets as nn


class Function:
    data_vars = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def delta(self):
        raise NotImplementedError

    def _unwrap_inputs(self, x):
        _x = nn.Variable(x) if isinstance(x, np.ndarray) else x
        assert isinstance(_x, nn.Variable), TypeError
        self.data_vars["in"] = _x
        return _x.data

    def _wrap_out(self, o):
        vo = nn.Variable(o)
        self.data_vars["out"] = vo
        return vo

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Transpose(Function):
    forward_axes = None

    def forward(self, x, axes=None):
        self.forward_axes = axes
        _x = self._unwrap_inputs(x)
        _o = np.transpose(_x, axes=axes)
        o = nn.Variable(_o)
        self.data_vars["out"] = o
        return o

    def delta(self):
        if self.forward_axes:
            axes = [self.forward_axes.index(i) for i in range(len(self.forward_axes))]
        else:
            axes = self.forward_axes
        return np.transpose(self.data_vars["out"].error, axes=axes)


class Reshape(Function):
    origin_shape = None

    def forward(self, x, newshape):
        self.origin_shape = x.shape
        _x = self._unwrap_inputs(x)
        _o = np.reshape(_x, newshape)
        o = self._wrap_out(_o)
        return o

    def delta(self):
        return np.reshape(self.data_vars["out"].error, self.origin_shape)


class Stack(Function):
    axis = None

    def _unwrap_inputs(self, x):
        _x = nn.Variable(x) if isinstance(x, np.ndarray) else x
        assert isinstance(_x, nn.Variable), TypeError
        if self.data_vars.get("in"):
            self.data_vars["in"].append(_x)
        else:
            self.data_vars["in"] = [_x]
        return _x.data

    def forward(self, x, axis=1):
        assert isinstance(x, (list, tuple)), TypeError
        self.axis = axis
        _x = [self._unwrap_inputs(data) for data in x]
        _o = np.stack(x, axis)
        o = self._wrap_out(_o)
        return o

    def delta(self):
        dz = self.data_vars["out"].error
        return np.split(dz, dz.shape[self.axis], axis=self.axis)


transpose = Transpose()
reshape = Reshape()
stack = Stack()


a = [np.ones((3,4)) for i in range(5)]
b = stack(a, 0)
b.set_error(np.ones((5, 3,4)))
c = reshape.delta()
print(len(c))