import neuralnets as nn
import numpy as np


class BaseRNN:
    def __init__(self,
                 n_in, n_hidden, n_out, n_layer, activation, batch_first,
                 w_initializer, b_initializer, use_bias):
        self.order = None
        self.name = None
        self._xi = None      # [n,t,i] if batch_first else [t,n,i]
        self._s = None      # [n,t,h] if batch_first else [t,n,h]

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_layer = n_layer  # TODO: add layer function
        self.batch_first = batch_first
        self.use_bias = use_bias
        self.data_vars = {}
        self.param_vars = {}
        self._batch_size = None
        if activation is None:
            self._a = nn.act.Linear()
        elif isinstance(activation, nn.act.Activation):
            self._a = activation
        else:
            raise TypeError
        self.wi = np.empty((n_in, n_hidden), dtype=np.float32)
        self.wh = np.empty((n_hidden, n_hidden), dtype=np.float32)
        self.wo = np.empty((n_hidden, n_out), dtype=np.float32)
        self.param_vars.update({"wi": self.wi, "wh": self.wh, "wo": self.wo})  # collect to param_vars
        self._wixb = None
        self._all_whsb = None   # [t,n,h]
        self._xo = None         # [t*n, h]
        self._woxb = None       # [t*n, o]

        if use_bias:
            self.bi = np.empty((1, n_in), dtype=np.float32)
            self.bh = np.empty((1, n_hidden), dtype=np.float32)
            self.bo = np.empty((1, n_out), dtype=np.float32)
            self.param_vars.update({"bi": self.bi, "bh": self.bh, "bo": self.bo})

        if w_initializer is None:
            w_initializer = nn.init.TruncatedNormal(0., 0.01)
        elif not isinstance(w_initializer, nn.init.BaseInitializer):
            raise TypeError
        [w_initializer.initialize(w) for w in [self.wi, self.wh, self.wo]]

        if use_bias:
            if b_initializer is None:
                b_initializer = nn.init.Constant(0.01)
            elif not isinstance(b_initializer, nn.init.BaseInitializer):
                raise TypeError
            [b_initializer.initialize(b) for b in [self.bh, self.bo]]

    def zero_state(self, batch_size):
        raise NotImplementedError

    def _process_inputs(self, x, s):
        raise NotImplementedError

    def _wrap_out(self, o, s):
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, x, state):
        return self.forward(x, state)


class RNN(BaseRNN):
    def __init__(self,
                 n_in,
                 n_hidden,
                 n_out,
                 n_layer=1,
                 activation=nn.act.tanh,
                 batch_first=False,
                 w_initializer=None,
                 b_initializer=None,
                 use_bias=True):
        super().__init__(
            n_in=n_in, n_hidden=n_hidden, n_out=n_out, n_layer=n_layer, activation=activation, batch_first=batch_first,
            w_initializer=w_initializer, b_initializer=b_initializer, use_bias=use_bias)

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.n_hidden), dtype=np.float32)

    def _process_inputs(self, x, s):
        # 0.check s
        # 1.unwarp variable
        # 2.get batch size
        # 3.set layer order
        assert x.ndim == 3, ValueError
        self._batch_size = x.shape[0] if self.batch_first else x.shape[1]
        if s is None:
            s = self.zero_state(batch_size=self._batch_size)
        if isinstance(x, np.ndarray):
            self.order = 0  # use layer input's information to set layer order
            x, s = x.astype(np.float32), s.astype(np.float32)
            if self.batch_first:
                x = x.transpose((1, 0, 2))     # => [t,n,i]
                s = s.transpose((1, 0, 2))     # => [t,n,h]
            # warp data to Variable
            x, s = nn.Variable(x), nn.Variable(s)
            x.info["new_layer_order"], s.info["new_layer_order"] = 0, 0

        self.data_vars["in"] = {"x": x, "s": s}     # all batch second/time step first
        self.order = max((x.info["new_layer_order"], s.info["new_layer_order"]))
        _x, _s = x.data, s.data
        return _x, _s

    def _wrap_out(self, o, s):
        if self.batch_first:
            o = o.transpose((1, 0, 2))      # => [n,t,o]
            s = s.transpose((1, 0, 2))      # => [n,t,h]
        o = nn.Variable(o)
        s = nn.Variable(s)
        o.info["new_layer_order"] = self.order + 1
        s.info["new_layer_order"] = self.order + 1
        self.data_vars["out"] = {"o": o, "s": s}
        return o, s

    def forward(self, x, initial_state=None):
        s = initial_state
        self._xi, self._s = self._process_inputs(x, s)
        print(self.order)
        _x_reshape = self._xi.reshape(-1, self.n_hidden)     # => [t*n, i]
        self._wixb = _x_reshape.dot(self.wi)   # => [t*n, h]
        if self.use_bias:
            self._wixb += self.bi
        self._wixb = self._wixb.transpose((-1, self._batch_size, self.n_hidden))    # => [t, n, h]
        all_hs = [self._s]
        for step_wix in self._wixb:
            all_hs.append(step_wix + all_hs[-1].dot(self.wh))   # => [n,h]
            if self.use_bias:
                all_hs[-1] += self.bh
            all_hs[-1][:] = self._a(all_hs[-1])
        self._all_whsb = np.vstack(all_hs)    # [t,n,h]
        
        self._xo = self._all_whsb.reshape(-1, self.n_hidden)     # [t*n, h]
        self._woxb = self._xo.dot(self.wo)    # => [t*n, o]
        if self.use_bias:
            self._woxb += self.bo
        out_reshape = self._woxb.reshape(-1, self._batch_size, self.n_out)     # [t,n,o]
        if self.batch_first:
            out_reshape = out_reshape.transpose((1, 0, 2))  # [n,t,o]
        wrapped_out, wrapped_s = self._wrap_out(out_reshape, s)
        return wrapped_out, wrapped_s

    def backward(self):
        dz = self.data_vars["out"]["o"].error
        if self.batch_first:
            dz = dz.transpose((-1, self._batch_size, self.n_out)).reshape(-1, self.n_out)   # => [t*n,o]
        grads = {"wo": self._xo.T.dot(dz)}      # [t*n,h].T @ [t*n,o] => [h, o]

        # initialize gradients tmp
        g_wh, g_wi = np.zeros_like(self.wh), np.zeros_like(self.wi)
        if self.use_bias:
            grads.update({"bo": np.sum(dz, axis=0, keepdims=True)})
            g_bh, g_bi = np.zeros_like(self.bh), np.zeros_like(self.bi)

        do = dz.dot(self.wo.T).reshape(-1, self._batch_size, self.n_hidden)  # [t*n, o] @ [o,h] => [t, n, h]
        dh = np.empty((self._batch_size, self.n_hidden), dtype=np.float32)     # [n, h]
        for t in range(do.shape[0]-1, -1, -1):
            do *= self._a.derivative(self._all_whsb[t+1])
            g_wh += self._all_whsb[t].T.dot(do[t])
            if self.use_bias:
                gb = np.sum(do[t], axis=0, keepdims=True)
                g_bh += gb
                g_bi += gb
            dh[:] = self.wh.T.dot(do[t])
            for back_t in range(t-1, -1, -1):
                g_wh += self._all_whsb[back_t].T.dot(dh)
                g_wi += self._xi[back_t].T.dot(dh)
                if self.use_bias:
                    gb = np.sum(dh, axis=0, keepdims=True)
                    g_bh += gb
                    g_bi += gb
                dh[:] = self.wh.T.dot(dh)

        grads.update({"wi": g_wi, "wh": g_wh})

        # dx, ds
        self.data_vars["in"]["s"].error[:] = dz.dot(self.wh.T)
        self.data_vars["in"]["x"].error[:] = dz.dot(self.wi.T)  # pass error to the layer before
        return grads

    s



if __name__ == "__main__":
    cell = RNNCell(n_in=3, n_hidden=4)
    x = np.arange(6).reshape(2,3)
    s = None
    for t in range(3):
        s = cell.forward(x, s)
        print(s)

    cell.data_vars["out"]["s"].error = np.ones_like(s)
    for t in range(3):
        grad = cell.backward()
