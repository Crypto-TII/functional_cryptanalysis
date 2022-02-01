"""
Copyright 2022 Technology Innovation Institute LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random
import stp


class XoodooState(object):
    nrows = 3
    ncols = 4
    word_size = 32

    def __init__(self, state):
        """
        Construct an instance of XoodooState

        INPUT:

        - ``state`` -- input state

        TESTS::

            >>> from xoodoo import XoodooState
            >>> S0 = XoodooState(range(12))
            >>> S0
            0x00000000 0x00000001 0x00000002 0x00000003
            0x00000004 0x00000005 0x00000006 0x00000007
            0x00000008 0x00000009 0x0000000A 0x0000000B
            >>> S1 = XoodooState([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
            >>> S1
            0x00000000 0x00000001 0x00000002 0x00000003
            0x00000004 0x00000005 0x00000006 0x00000007
            0x00000008 0x00000009 0x0000000A 0x0000000B
            >>> S2 = XoodooState([1])
            Traceback (most recent call last):
            ...
            TypeError: invalid input format
        """
        nrows = XoodooState.nrows
        ncols = XoodooState.ncols
        self._state = []
        if len(state) == nrows * ncols:
            self._state = [[int(state[i * ncols + j]) for j in range(ncols)] for i in range(nrows)]
        elif len(state) == nrows and all(len(row) == ncols for row in state):
            self._state = [[int(state[i][j]) for j in range(ncols)] for i in range(nrows)]
        else:
            raise TypeError("invalid input format")

    @property
    def state(self):
        """
        Return the Xoodoo state

        TESTS::

            >>> from xoodoo import XoodooState
            >>> S = XoodooState(range(12))
            >>> S.state
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        """
        return self._state

    def hamming_weight(self):
        """
        Return the Hamming weight of the state

        TESTS::

            >>> from xoodoo import XoodooState
            >>> S = XoodooState(range(12))
            >>> S.hamming_weight()
            20
        """
        weight = 0
        for i in range(XoodooState.nrows):
            for j in range(XoodooState.ncols):
                for k in range(XoodooState.word_size):
                    weight += (self[i][j] & (1 << k)) >> k

        return weight

    def trail_weight(self):
        """
        Return the trail weight of the state

        TESTS::

            >>> from xoodoo import XoodooState
            >>> S = XoodooState(range(12))
            >>> S.trail_weight()
            24
        """
        def hamming_weight(x):
            hw = 0
            for k in range(XoodooState.word_size):
                hw += (x & (1 << k)) >> k
            return hw

        s = self.state
        w = 0
        for j in range(XoodooState.ncols):
            w += hamming_weight(s[0][j] | s[1][j] | s[2][j])
        return 2*w

    def __repr__(self):
        """
        Return the string representation of XoodooState

        TESTS::

            >>> from xoodoo import XoodooState
            >>> S = XoodooState([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            >>> S
            0x00000000 0x00000001 0x00000002 0x00000003
            0x00000004 0x00000005 0x00000006 0x00000007
            0x00000008 0x00000009 0x0000000A 0x0000000B
        """
        nrows = XoodooState.nrows
        ncols = XoodooState.ncols

        hex_str = [["0x" + hex(self[i][j])[2:].zfill(8).upper() for j in range(ncols)] for i in range(nrows)]

        return '\n'.join([' '.join(hex_str[i]) for i in range(nrows)])

    def __getitem__(self, r):
        """
        INPUT:

        - ``r`` -- row index

        TESTS::

            >>> from xoodoo import XoodooState
            >>> S = XoodooState(range(12))
            >>> S[0]
            [0, 1, 2, 3]
        """
        return self.state[r]

    def __setitem__(self, r, v):
        """
        Replace the value of row `r` with vector `v`

        INPUT:

        - ``r`` -- row index
        - ``v`` -- a list/tuple

        TESTS::

            >>> from xoodoo import XoodooState
            >>> S = XoodooState(range(12))
            >>> S
            0x00000000 0x00000001 0x00000002 0x00000003
            0x00000004 0x00000005 0x00000006 0x00000007
            0x00000008 0x00000009 0x0000000A 0x0000000B
            >>> S[0] = [0xFFFF0000, 0x0000FFFF, 0x00FF00FF00, 0x0F0F0F0F]
            >>> S
            0xFFFF0000 0x0000FFFF 0xFF00FF00 0x0F0F0F0F
            0x00000004 0x00000005 0x00000006 0x00000007
            0x00000008 0x00000009 0x0000000A 0x0000000B
        """
        if len(v) != XoodooState.ncols:
            raise TypeError("the length of v must be equal to %d" % XoodooState.ncols)

        self.state[r] = v


class XoodooTrail(object):

    _nsteps_in_a_round = 4

    def __init__(self, trail):
        """
        Construct an instance of Xoodoo trail

        INPUT:

        - ``trail`` -- a list of Xoodoo state
        """
        self._trail = [XoodooState(state) for state in trail]
        self._nrounds = len(trail) // XoodooTrail._nsteps_in_a_round

    @property
    def trail(self):
        """
        Return the trail
        """
        return self._trail

    @property
    def nrounds(self):
        """
        Return the number of rounds covered for the given trail
        """
        return self._nrounds

    @staticmethod
    def input_round_index(r):
        """
        Return the index of input state at round `r` in the trail array

        INPUT:

        - ``r`` -- round index
        """
        nsteps = XoodooTrail._nsteps_in_a_round
        return nsteps * r

    def input(self):
        """
        Return the input state of the trail
        """
        return self.input_round(0)

    def output(self):
        """
        Return the output state of the trail
        """
        return self.output_round(self.nrounds - 1)

    def input_round(self, r):
        """
        Return the input state at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.trail[XoodooTrail.input_round_index(r)]

    def input_theta(self, r):
        """
        Return the input state of the theta function at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.input_round(r)

    def input_rho_west(self, r):
        """
        Return the input state of the rho west function at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.trail[XoodooTrail.input_round_index(r) + 1]

    def input_chi(self, r):
        """
        Return the input state of the chi function at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.trail[XoodooTrail.input_round_index(r) + 2]

    def input_rho_east(self, r):
        """
        Return the input state of the rho east function at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.trail[XoodooTrail.input_round_index(r) + 3]

    def output_round(self, r):
        """
        Return the output state of round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.input_round(r + 1)

    def output_theta(self, r):
        """
        Return the output state of theta function at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.input_rho_west(r)

    def output_rho_west(self, r):
        """
        Return the output state of the rho west function at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.input_chi(r)

    def output_chi(self, r):
        """
        Return the output state of the chi function at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.input_rho_east(r)

    def output_rho_east(self, r):
        """
        Return the output state of the rho east function at round `r`

        INPUT:

        - ``r`` -- round index
        """
        return self.output_round(r)

    def weight(self):
        """
        Return the weight of the trail
        """
        w = sum(self.input_chi(r).trail_weight() for r in range(self.nrounds))
        return int(w)


class Xoodoo(object):
    max_nrounds = 12
    round_constants = [0x00000058, 0x00000038, 0x000003C0, 0x000000D0, 0x00000120, 0x00000014,
                       0x00000060, 0x0000002C, 0x00000380, 0x000000F0, 0x000001A0, 0x00000012]
    _input_round_var_name = 'x'

    _input_chi_var_name = 's'
    _output_chi_var_name = 't'

    _aux_theta_var_name = 'p'
    _output_theta_var_name = 'e'

    _diff_input_round_var_name = _input_round_var_name.upper()
    _diff_input_chi_var_name = _input_chi_var_name.upper()
    _diff_output_chi_var_name = _output_chi_var_name.upper()
    _diff_aux_theta_var_name = _aux_theta_var_name.upper()
    _diff_output_theta_var_name = _output_theta_var_name.upper()

    _weight_var_name = 'w'
    _total_weight_var_name = _weight_var_name.upper()

    def __init__(self, nrounds):
        """
        Construct an instance of Xoodoo

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=4)
            >>> X
            4-rounds of Xoodoo
        """
        if not 1 <= nrounds <= 12:
            raise ValueError("nrounds must be between 1 <= nrounds <= 12")

        self._nrounds = nrounds
        self._solver = stp.Solver()

    @property
    def nrounds(self):
        """
        Return the number of rounds

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=4)
            >>> X.nrounds
            4
        """
        return self._nrounds

    @property
    def solver(self):
        """
        Return the STP solver

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=4)
            >>> X.solver # doctest: +ELLIPSIS
            <stp.stp.Solver object at 0x...>
        """
        return self._solver

    def round_constant(self, r):
        """
        Return the round constant at round `r`

        INPUT:

        - ``r`` -- round index

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=4)
            >>> "0x" + hex(X.round_constant(0))[2:].zfill(8)
            '0x00000380'

        TESTS::

            >>> X.round_constant(0) == Xoodoo.round_constants[-X.nrounds:][0]
            True
        """
        if not 0 <= r < self.nrounds:
            raise ValueError("r must be in the range 0 <= r < %d" % self.nrounds)
        return Xoodoo.round_constants[-self.nrounds:][r]

    def rotate_left_constraint(self, x, y, r):
        """
        Return the bitwise left-rotation constraint

        INPUT:

        - ``x`` -- input word
        - ``y`` -- output word
        - ``r`` -- rotation constant

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=4)
            >>> S = X.solver
            >>> a = S.bitvec('a', width=32)
            >>> b = S.bitvec('b', width=32)
            >>> S.add(X.rotate_left_constraint(a, b, 2))
            >>> S.add(a == 0x80000000)
            >>> S.check()
            True
            >>> S.model(b.name)
            2L
        """
        if not isinstance(y, stp.Expr):
            raise TypeError("y must be an instance of stp.stp.Expr")

        if y.width != XoodooState.word_size:
            raise ValueError("the width of y must be equal to %d" % XoodooState.word_size)

        return Xoodoo.rotate_left(x, r) == y

    @staticmethod
    def rotate_left(x, r):
        """
        Return bitwise left-rotation on `x` with `r`-bit rotation

        INPUT:

        - ``x`` -- input word
        - ``r`` -- rotation constant

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=4)
            >>> S = X.solver
            >>> a = S.bitvec('a', width=32)
            >>> b = S.bitvec('b', width=32)
            >>> S.add(X.rotate_left(a, 2) == b)
            >>> S.add(a == 0x80000000)
            >>> S.check()
            True
            >>> S.model(b.name)
            2L
        """
        if not isinstance(r, (int, long)):
            raise TypeError("r must be an int or long")

        if isinstance(x, stp.Expr) and x.width != XoodooState.word_size:
            raise ValueError("the width of x must be equal to %d" % XoodooState.word_size)

        if not 0 <= r < XoodooState.word_size:
            raise ValueError("r must be in the range 0 <= r < %d" % XoodooState.word_size)

        val = ((x << r) | (x >> (XoodooState.word_size - r)))
        if isinstance(x, (int, long)):
            val &= 0xFFFFFFFF

        return val

    def input_round_varstrs(self, r):
        """
        Return input variables string for round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=4)
            >>> X.input_round_varstrs(0) # doctest: +NORMALIZE_WHITESPACE
            [['x000000', 'x000001', 'x000002', 'x000003'],
            ['x000100', 'x000101', 'x000102', 'x000103'],
            ['x000200', 'x000201', 'x000202', 'x000203']]

        TESTS::

            >>> X.input_round_varstrs(-1)
            Traceback (most recent call last):
            ...
            ValueError: r must be in the range 0 <= r <= 4
            >>> X.input_round_varstrs(5)
            Traceback (most recent call last):
            ...
            ValueError: r must be in the range 0 <= r <= 4
        """
        return self.varstrs(r, var_name=Xoodoo._input_round_var_name)

    def input_round_vars(self, r):
        """
        Return input variables for round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> x = xoodoo.input_round_vars(0)
            >>> [[v.name for v in row] for row in x] # doctest: +NORMALIZE_WHITESPACE
            [['x000000', 'x000001', 'x000002', 'x000003'],
            ['x000100', 'x000101', 'x000102', 'x000103'],
            ['x000200', 'x000201', 'x000202', 'x000203']]
        """
        return self.vars(r, var_name=Xoodoo._input_round_var_name)

    def output_round_varstrs(self, r):
        """
        Return output variables string of round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=4)
            >>> X.output_round_varstrs(0) # doctest: +NORMALIZE_WHITESPACE
            [['x010000', 'x010001', 'x010002', 'x010003'],
            ['x010100', 'x010101', 'x010102', 'x010103'],
            ['x010200', 'x010201', 'x010202', 'x010203']]
        """
        return self.input_round_varstrs(r + 1)

    def output_round_vars(self, r):
        """
        Return output variables of round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> x = xoodoo.output_round_vars(0)
            >>> [[ v.name for v in row] for row in x]  # doctest: +NORMALIZE_WHITESPACE
            [['x010000', 'x010001', 'x010002', 'x010003'],
            ['x010100', 'x010101', 'x010102', 'x010103'],
            ['x010200', 'x010201', 'x010202', 'x010203']]
        """
        return self.input_round_vars(r + 1)

    def aux_vars(self, r, var_name):
        """
        Return a list of auxilliary variables

        INPUT:

        - ``r`` -- round number
        - ``var_name`` -- variable name
        """
        x = ["%s%02d%02d" % (var_name, r, j) for j in range(XoodooState.ncols)]
        return [self.solver.bitvec(x[j], width=XoodooState.word_size) for j in range(XoodooState.ncols)]

    def vars(self, r, var_name):
        """
        Return a list of variables

        INPUT:

        - ``r`` -- round number
        - ``var_name`` -- variable name
        """
        x = self.varstrs(r, var_name)
        return [[self.solver.bitvec(x[i][j], width=XoodooState.word_size)
                 for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)]

    @staticmethod
    def is_valid_state_format(state):
        """
        Return `True` if `l` is a `3 \times 4` list/tuple

        INPUT:

        - ``l`` -- a list/tuple

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> Xoodoo.is_valid_state_format([])
            False
            >>> Xoodoo.is_valid_state_format([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
            True
        """
        return len(state) == XoodooState.nrows and all(len(row) == XoodooState.ncols for row in state)

    def varstrs(self, r, var_name):
        """
        Return a list of variable in string format

        INPUT:

        - ``r`` -- round number
        - ``var_name`` -- name of variable

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.varstrs(0, var_name='x') # doctest: +NORMALIZE_WHITESPACE
            [['x000000', 'x000001', 'x000002', 'x000003'],
            ['x000100', 'x000101', 'x000102', 'x000103'],
            ['x000200', 'x000201', 'x000202', 'x000203']]
            >>> xoodoo.varstrs(1, var_name='y') # doctest: +NORMALIZE_WHITESPACE
            [['y010000', 'y010001', 'y010002', 'y010003'],
            ['y010100', 'y010101', 'y010102', 'y010103'],
            ['y010200', 'y010201', 'y010202', 'y010203']]
        """
        if not 0 <= r <= self.nrounds:
            raise ValueError("r must be in the range 0 <= r <= %d" % self.nrounds)
        return [['%s%02d%02d%02d' % (var_name, r, i, j) for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)]

    def input_chi_varstrs(self, r):
        """
        Return a list of input variables in string format for chi at round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.input_chi_varstrs(0) # doctest: +NORMALIZE_WHITESPACE
            [['s000000', 's000001', 's000002', 's000003'],
            ['s000100', 's000101', 's000102', 's000103'],
            ['s000200', 's000201', 's000202', 's000203']]
        """
        return self.varstrs(r, Xoodoo._input_chi_var_name)

    def input_chi_vars(self, r):
        """
        Return a list of input variables for chi at round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> x = xoodoo.input_chi_vars(0)
            >>> [[v.name for v in row] for row in x] # doctest: +NORMALIZE_WHITESPACE
            [['s000000', 's000001', 's000002', 's000003'],
            ['s000100', 's000101', 's000102', 's000103'],
            ['s000200', 's000201', 's000202', 's000203']]
        """
        return self.vars(r, Xoodoo._input_chi_var_name)

    def output_chi_varstrs(self, r):
        """
        Return a list of output variables in string format for chi at round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.output_chi_varstrs(0) # doctest: +NORMALIZE_WHITESPACE
            [['t000000', 't000001', 't000002', 't000003'],
            ['t000100', 't000101', 't000102', 't000103'],
            ['t000200', 't000201', 't000202', 't000203']]
        """
        return self.varstrs(r, Xoodoo._output_chi_var_name)

    def output_chi_vars(self, r):
        """
        Return a list of input variables for chi at round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> x = xoodoo.output_chi_vars(0)
            >>> [[v.name for v in row] for row in x] # doctest: +NORMALIZE_WHITESPACE
            [['t000000', 't000001', 't000002', 't000003'],
            ['t000100', 't000101', 't000102', 't000103'],
            ['t000200', 't000201', 't000202', 't000203']]
        """
        return self.vars(r, Xoodoo._output_chi_var_name)

    def input_theta_vars(self, r):
        """
        Return a list of input variables for theta at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.input_theta_vars(0) == xoodoo.input_round_vars(0)
            True
        """
        return self.input_round_vars(r)

    def aux_theta_vars(self, r):
        """
        Return a list of auxilliary variables for theta at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> x = xoodoo.aux_theta_vars(0)
            >>> [v.name for v in x]
            ['p0000', 'p0001', 'p0002', 'p0003']
        """
        return self.aux_vars(r, Xoodoo._aux_theta_var_name)

    def output_theta_vars(self, r):
        """"
        Return a list of output variables for theta at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=3)
            >>> x = xoodoo.output_theta_vars(0)
            >>> [[v.name for v in row] for row in x] # doctest: +NORMALIZE_WHITESPACE
            [['e000000', 'e000001', 'e000002', 'e000003'],
            ['e000100', 'e000101', 'e000102', 'e000103'],
            ['e000200', 'e000201', 'e000202', 'e000203']]
        """
        return self.vars(r, Xoodoo._output_theta_var_name)

    def theta_constraints(self, r):
        """
        Return a list of constraints for theta at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> solver = xoodoo.solver
            >>> for constraint in xoodoo.theta_constraints(0):
            ...     solver.add(constraint)
            >>> randval = xoodoo.random_state()
            >>> for constraint in xoodoo.assignment_constraints(xoodoo.input_theta_vars(0), randval):
            ...     solver.add(constraint)
            >>> solver.check()
            True
            >>> e = xoodoo.output_theta_vars(0)
            >>> result = [[solver.model(e[i][j].name) for j in range(XoodooState.ncols)]
            ...            for i in range(XoodooState.nrows)]
            >>> def theta(x):
            ...     p = [x[0][j] ^ x[1][j] ^ x[2][j] for j in range(XoodooState.ncols)]
            ...     y = [[0 for __ in range(XoodooState.ncols)] for _ in range(XoodooState.nrows)]
            ...     for i in range(XoodooState.nrows):
            ...         for j in range(XoodooState.ncols):
            ...             k = (j + XoodooState.nrows) % XoodooState.ncols
            ...             y[i][j] = x[i][j] ^ xoodoo.rotate_left(p[k], 5) ^ xoodoo.rotate_left(p[k], 14)
            ...     return y
            >>> result == theta(randval)
            True
        """
        x = self.input_theta_vars(r)
        p = self.aux_theta_vars(r)
        y = self.output_theta_vars(r)

        return self._theta_constraints_(x, p, y)

    def _theta_constraints_(self, x, p, y):
        """
        Return a list of constraints for theta

        INPUT:

        - ``x`` -- input variables
        - ``p`` -- auxiliary variables
        - ``y`` -- output variables
        """
        constraints = [p[j] == x[0][j] ^ x[1][j] ^ x[2][j] for j in range(XoodooState.ncols)]
        for i in range(XoodooState.nrows):
            for j in range(XoodooState.ncols):
                k = (j + XoodooState.nrows) % XoodooState.ncols
                constraints.append(y[i][j] == (x[i][j] ^ Xoodoo.rotate_left(p[k], 5) ^ Xoodoo.rotate_left(p[k], 14)))

        return constraints

    def random_state(self):
        """
        Return a random xoodoo state

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=3)
            >>> state = xoodoo.random_state()
            >>> xoodoo.is_valid_state_format(state)
            True
        """
        return [[random.randint(0, 2**XoodooState.word_size) for __ in range(XoodooState.ncols)]
                for _ in range(XoodooState.nrows)]

    def assignment_constraints(self, state, value):
        """
        Return a list of constraints for assignment

        INPUT:

        - ``state`` -- variables representing Xoodoo state
        - ``value`` -- substituted value

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> x = xoodoo.input_round_vars(0)
            >>> v = xoodoo.random_state()
            >>> solver = xoodoo.solver
            >>> for constraint in xoodoo.assignment_constraints(x, v):
            ...     solver.add(constraint)
            >>> solver.check()
            True
            >>> all(solver.model(x[i][j].name) == v[i][j] for i in range(XoodooState.nrows)
            ...     for j in range(XoodooState.ncols))
            True
        """
        if not self.is_valid_state_format(state):
            raise TypeError("state must be an instance of %dx%d array of %d-bit words" %
                            (XoodooState.nrows, XoodooState.ncols, XoodooState.word_size))

        if not self.is_valid_state_format(value):
            raise TypeError("value must be an instance of %dx%d array of %d-bit words" %
                            (XoodooState.nrows, XoodooState.ncols, XoodooState.word_size))

        return [state[i][j] == value[i][j] for i in range(XoodooState.nrows) for j in range(XoodooState.ncols)]

    def chi_constraints(self, r):
        """
        Return a list of constraints for chi at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> import random
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> solver = xoodoo.solver
            >>> for constraint in xoodoo.chi_constraints(0):
            ...     solver.add(constraint)
            >>> randval = xoodoo.random_state()
            >>> s = xoodoo.input_chi_vars(0)
            >>> for constraint in xoodoo.assignment_constraints(s, randval):
            ...     solver.add(constraint)
            >>> solver.check()
            True
            >>> t = xoodoo.output_chi_vars(0)
            >>> result = [[solver.model(t[i][j].name) for j in range(XoodooState.ncols)]
            ...            for i in range(XoodooState.nrows)]
            >>> all(result[i][j] ==
            ...     randval[i][j] ^ (~randval[(i + 1) % XoodooState.nrows][j] & randval[(i + 2) % XoodooState.nrows][j])
            ...     for i in range(XoodooState.nrows) for j in range(XoodooState.ncols))
            True
        """
        x = self.input_chi_vars(r)
        y = self.output_chi_vars(r)

        return [y[i][j] == x[i][j] ^ (~x[(i + 1) % XoodooState.nrows][j] & x[(i + 2) % XoodooState.nrows][j])
                 for j in range(XoodooState.ncols) for i in range(XoodooState.nrows)]

    def rho_east_constraints(self, r):
        """
        Return a list of constraints for rho east at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> import random
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> solver = xoodoo.solver
            >>> for constraint in xoodoo.rho_east_constraints(0):
            ...     solver.add(constraint)
            >>> randval = xoodoo.random_state()
            >>> out = []
            >>> out.append( [randval[0][j] for j in range(XoodooState.ncols)] )
            >>> out.append( [xoodoo.rotate_left(randval[1][j], 1) for j in range(XoodooState.ncols)] )
            >>> out.append( [xoodoo.rotate_left(randval[2][(j + 2) % XoodooState.ncols], 8) for j in range(XoodooState.ncols)] )
            >>> x = xoodoo.output_chi_vars(0)
            >>> for constraint in xoodoo.assignment_constraints(x, randval):
            ...     solver.add(constraint)
            >>> solver.check()
            True
            >>> y = xoodoo.output_round_vars(0)
            >>> result = [[solver.model(y[i][j].name) for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)]
            >>> all(result[i][j] == out[i][j] for i in range(XoodooState.nrows) for j in range(XoodooState.ncols))
            True
        """
        x = self.output_chi_vars(r)
        y = self.output_round_vars(r)

        return self._rho_east_constraints_(x, y)

    def _rho_east_constraints_(self, x, y):
        """
        Return a list of constraints for rho east

        INPUT:

        - ``x`` -- input variables
        - ``y`` -- output variables
        """
        constraints = [y[0][j] == x[0][j] for j in range(XoodooState.ncols)]
        constraints += [y[1][j] == Xoodoo.rotate_left(x[1][j], 1) for j in range(XoodooState.ncols)]
        constraints += [y[2][j] == Xoodoo.rotate_left(x[2][(j + 2) % XoodooState.ncols], 8)
                        for j in range(XoodooState.ncols)]

        return constraints

    def rho_west_and_iota_constraints(self, r):
        """
        Return a list of constraints for composition of rho west and iota at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> solver = xoodoo.solver
            >>> for constraint in xoodoo.rho_west_and_iota_constraints(0):
            ...     solver.add(constraint)
            >>> randval = xoodoo.random_state()
            >>> for constraint in xoodoo.assignment_constraints(xoodoo.output_theta_vars(0), randval):
            ...     solver.add(constraint)
            >>> def theta_and_iota(x):
            ...     y = list()
            ...     y.append([x[0][0] ^ xoodoo.round_constant(0)] + x[0][1:])
            ...     y.append([x[1][3]] + x[1][:3])
            ...     y.append([xoodoo.rotate_left(x[2][j], 11) for j in range(XoodooState.ncols)])
            ...     return y
            >>> solver.check()
            True
            >>> out_vars = xoodoo.input_chi_vars(0)
            >>> result = [[solver.model(out_vars[i][j].name) for j in range(XoodooState.ncols)]
            ...            for i in range(XoodooState.nrows)]
            >>> result == theta_and_iota(randval)
            True
        """
        x = self.output_theta_vars(r)
        y = self.input_chi_vars(r)

        constraints = self._rho_west_constraints_(x, y, r)

        return constraints

    def _rho_west_constraints_(self, x, y, r=None):
        """
        Return a list of constraints for rho west

        If the round number `r` is specified, then the round constant is included in the constraints

        INPUT:

        - ``x`` -- input variables
        - ``y`` -- output variables
        - ``r`` -- round number (default: None)
        """
        constraints = []
        if r is not None:
            constraints += [y[0][0] == x[0][0] ^ self.round_constant(r)]
            constraints += [y[0][j] == x[0][j] for j in range(1, XoodooState.ncols)]
        else:
            constraints += [y[0][j] == x[0][j] for j in range(XoodooState.ncols)]

        constraints += [y[1][j] == x[1][(j + XoodooState.nrows) % XoodooState.ncols] for j in range(XoodooState.ncols)]
        constraints += [y[2][j] == Xoodoo.rotate_left(x[2][j], 11) for j in range(XoodooState.ncols)]

        return constraints

    def round_constraints(self, r):
        """
        Return a list of constraints of the `r`-th round

        INPUT:

        - ``r`` -- round number
        """
        return self.theta_constraints(r) + self.rho_west_and_iota_constraints(r) + self.chi_constraints(r) +\
               self.rho_east_constraints(r)

    def permutation_constraints(self):
        """
        Return a list of constraints for the Xoodoo permutation

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=12)
            >>> solver = xoodoo.solver
            >>> for constraint in xoodoo.permutation_constraints():
            ...     solver.add(constraint)
            >>> x = xoodoo.input_variables()
            >>> for constraint in xoodoo.assignment_constraints(x, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]):
            ...     solver.add(constraint)
            >>> solver.check()
            True
            >>> y = xoodoo.output_variables()
            >>> out_value = [[solver.model(y[i][j].name) for j in range(XoodooState.ncols)]
            ...               for i in range(XoodooState.nrows)]
            >>> correct_output = [
            ...     [2312493197, 2841902271, 455290137, 4289044500],
            ...     [917602566, 2949104126, 2934275262, 2809479357],
            ...     [780593264, 4277516233, 2337254898, 1582252130]
            ... ]
            >>> out_value == correct_output
            True
            >>> xoodoo_3rounds = Xoodoo(nrounds=3)
            >>> for constraint in xoodoo_3rounds.permutation_constraints():
            ...     xoodoo_3rounds.solver.add(constraint)
            >>> x = xoodoo_3rounds.input_variables()
            >>> for constraint in xoodoo.assignment_constraints(x, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]):
            ...     xoodoo_3rounds.solver.add(constraint)
            >>> xoodoo_3rounds.solver.check()
            True
            >>> y = xoodoo_3rounds.output_variables()
            >>> out_value = [[xoodoo_3rounds.solver.model(y[i][j].name) for j in range(XoodooState.ncols)]
            ...               for i in range(XoodooState.nrows)]
            >>> correct_output = [
            ...     [3367309476, 2795523367, 3279790372, 1225296034],
            ...     [2440626244, 4167606016, 210768320, 1157228578],
            ...     [2173818853, 1684836713, 160556720, 1812112827]
            ... ]
            >>> out_value == correct_output
            True
        """
        return sum([self.round_constraints(r) for r in range(self.nrounds)], [])

    def input_variables(self):
        """
        Return a list of input variables of the permutation

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> in_vars = xoodoo.input_variables()
            >>> [[in_vars[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)] # doctest: +NORMALIZE_WHITESPACE
            [['x000000', 'x000001', 'x000002', 'x000003'],
            ['x000100', 'x000101', 'x000102', 'x000103'],
            ['x000200', 'x000201', 'x000202', 'x000203']]
        """
        return self.input_round_vars(0)

    def diff_input_round_vars(self, r):
        """
        Return a list of input round difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> d_in_vars = xoodoo.diff_input_round_vars(0)
            >>> [[d_in_vars[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)]  # doctest: +NORMALIZE_WHITESPACE
            [['X000000', 'X000001', 'X000002', 'X000003'],
            ['X000100', 'X000101', 'X000102', 'X000103'],
            ['X000200', 'X000201', 'X000202', 'X000203']]
        """
        return self.vars(r, Xoodoo._diff_input_round_var_name)

    def diff_output_round_vars(self, r):
        """
        Return a list of output round difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> d_out_vars = xoodoo.diff_output_round_vars(0)
            >>> [[d_out_vars[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)]  # doctest: +NORMALIZE_WHITESPACE
            [['X010000', 'X010001', 'X010002', 'X010003'],
            ['X010100', 'X010101', 'X010102', 'X010103'],
            ['X010200', 'X010201', 'X010202', 'X010203']]
        """
        return self.diff_input_round_vars(r + 1)

    def diff_input_theta_vars(self, r):
        """
        Return a list of input theta difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo, XoodooState
            >>> xoodoo = Xoodoo(nrounds=3)
            >>> d = xoodoo.diff_input_theta_vars(0)
            >>> d_in_round_vars = xoodoo.diff_input_round_vars(0)
            >>> all(d[i][j].name == d_in_round_vars[i][j].name
            ...     for i in range(XoodooState.nrows)
            ...     for j in range(XoodooState.ncols))
            True
        """
        return self.diff_input_round_vars(r)

    def diff_aux_theta_vars(self, r):
        """
        Return a list of auxiliary theta difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> d_aux_theta_vars = xoodoo.diff_aux_theta_vars(0)
            >>> [var_.name for var_ in d_aux_theta_vars]
            ['P0000', 'P0001', 'P0002', 'P0003']
        """
        return self.aux_vars(r, Xoodoo._diff_aux_theta_var_name)

    def diff_output_theta_vars(self, r):
        """
        Return a list of output theta difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> d_out_theta_vars = xoodoo.diff_output_theta_vars(0)
            >>> [[d_out_theta_vars[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)] # doctest: +NORMALIZE_WHITESPACE
            [['E000000', 'E000001', 'E000002', 'E000003'],
            ['E000100', 'E000101', 'E000102', 'E000103'],
            ['E000200', 'E000201', 'E000202', 'E000203']]
        """
        return self.vars(r, Xoodoo._diff_output_theta_var_name)

    def diff_theta_constraints(self, r):
        """
        Return a list of constraints for the differential propagation of theta at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> constraints = xoodoo.diff_theta_constraints(0)
            >>> for constraint in constraints:
            ...     xoodoo.solver.add(constraint)
            >>> xoodoo.solver.check()
            True
        """
        x = self.diff_input_theta_vars(r)
        p = self.diff_aux_theta_vars(r)
        y = self.diff_output_theta_vars(r)

        return self._theta_constraints_(x, p, y)

    def diff_input_chi_vars(self, r):
        """
        Return a list of input chi difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> d_in_chi_vars = xoodoo.diff_input_chi_vars(0)
            >>> [[d_in_chi_vars[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)] # doctest: +NORMALIZE_WHITESPACE
            [['S000000', 'S000001', 'S000002', 'S000003'],
            ['S000100', 'S000101', 'S000102', 'S000103'],
            ['S000200', 'S000201', 'S000202', 'S000203']]
        """
        return self.vars(r, Xoodoo._diff_input_chi_var_name)

    def diff_output_chi_vars(self, r):
        """
        Return a list of output chi difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> d_out_chi_vars = xoodoo.diff_output_chi_vars(0)
            >>> [[d_out_chi_vars[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)] # doctest: +NORMALIZE_WHITESPACE
            [['T000000', 'T000001', 'T000002', 'T000003'],
            ['T000100', 'T000101', 'T000102', 'T000103'],
            ['T000200', 'T000201', 'T000202', 'T000203']]
        """
        return self.vars(r, Xoodoo._diff_output_chi_var_name)

    def diff_input_rho_west_vars(self, r):
        """
        Return a list of input rho west difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.diff_input_rho_west_vars(0) == xoodoo.diff_output_theta_vars(0)
            True
        """
        return self.diff_output_theta_vars(r)

    def diff_output_rho_west_vars(self, r):
        """
        Return a list of output rho west difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.diff_output_rho_west_vars(0) == xoodoo.diff_input_chi_vars(0)
            True
        """
        return self.diff_input_chi_vars(r)

    def diff_rho_west_constraints(self, r):
        """
        Return a list of constraints for the differential propagation of rho west at round `r`

        INPUT:

        - ``r`` -- round number
        """
        x = self.diff_input_rho_west_vars(r)
        y = self.diff_output_rho_west_vars(r)

        return self._rho_west_constraints_(x, y)

    def diff_input_rho_east_vars(self, r):
        """
        Return a list of input rho east difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.diff_input_rho_east_vars(0) == xoodoo.diff_output_chi_vars(0)
            True
        """
        return self.diff_output_chi_vars(r)

    def diff_output_rho_east_vars(self, r):
        """
        Return a list of output rho east difference variables at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.diff_output_rho_east_vars(0) == xoodoo.diff_output_round_vars(0)
            True
        """
        return self.diff_output_round_vars(r)

    def diff_rho_east_constraints(self, r):
        """
        Return a list of constraints for the differential propagation of rho east at round `r`

        INPUT:

        - ``r`` -- round number
        """
        x = self.diff_input_rho_east_vars(r)
        y = self.diff_output_rho_east_vars(r)

        return self._rho_east_constraints_(x, y)

    def diff_chi_constraints(self, r):
        """
        Return a list of constraints for the differential propagation of chi at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> constraints = xoodoo.diff_chi_constraints(0)
            >>> for constraint in constraints:
            ...     xoodoo.solver.add(constraint)
            >>> xoodoo.solver.check()
            True
        """
        x = self.diff_input_chi_vars(r)
        y = self.diff_output_chi_vars(r)
        w = self.weight_vars(r)

        constraints = []
        for j in range(XoodooState.ncols):
            constraints += self._diff_chi_wordwise_constraints_(x[0][j], x[1][j], x[2][j], y[0][j], y[1][j], y[2][j])
            constraints += [self.hamming_weight_constraint(x[0][j] | x[1][j] | x[2][j], w[j])]

        return constraints

    def _diff_chi_wordwise_constraints_(self, dx, dy, dz, dxp, dyp, dzp):
        """
        Return a list of constraints for the word-wise differential propagation of chi

        INPUT:

        - ``dx, dy, dz`` -- input difference
        - ``dxp, dyp, dzp`` -- output difference

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> chi_ddt = [[0], [1, 3, 5, 7], [2, 3, 6, 7], [1, 2, 5, 6], [4, 5, 6, 7], [1, 3, 4, 6], [2, 3, 4, 5],
            ...            [1, 2, 4, 7]]
            >>> is_correct = []
            >>> for input_diff in range(len(chi_ddt)):
            ...     for output_diff in chi_ddt[input_diff]:
            ...         xoodoo = Xoodoo(nrounds=4)
            ...         solver = xoodoo.solver
            ...         dx, dy, dz = solver.bitvecs('dx dy dz', width=1)
            ...         dxp, dyp, dzp = solver.bitvecs('dxp dyp dzp', width=1)
            ...         in_diff = map(int, bin(input_diff)[2:].zfill(3))
            ...         out_diff = map(int, bin(output_diff)[2:].zfill(3))
            ...         solver.add(dx == in_diff[0])  #zero is the most-significant bit
            ...         solver.add(dy == in_diff[1])
            ...         solver.add(dz == in_diff[2])
            ...         solver.add(dxp == out_diff[0])
            ...         solver.add(dyp == out_diff[1])
            ...         solver.add(dzp == out_diff[2])
            ...         is_correct.append(solver.check())
            >>> all(is_correct)
            True
        """
        constraints = [
            (~dx & ~dy & ~dz) & (dxp | dyp | dzp) == 0,
            (~dx & ~dy &  dz) & ~dzp == 0,
            (~dx &  dy & ~dz) & ~dyp == 0,
            ( dx & ~dy & ~dz) & ~dxp == 0,
            (~dx &  dy &  dz) & ~(dyp ^ dzp) == 0,
            ( dx & ~dy &  dz) & ~(dxp ^ dzp) == 0,
            ( dx &  dy & ~dz) & ~(dxp ^ dyp) == 0,
            ( dx &  dy &  dz) & ~(dxp ^ dyp ^ dzp) == 0,
        ]

        return constraints

    def output_variables(self):
        """
        Return a list of output variables of the permutation

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> out_vars = xoodoo.output_variables()
            >>> [[out_vars[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)] # doctest: +NORMALIZE_WHITESPACE
            [['x040000', 'x040001', 'x040002', 'x040003'],
            ['x040100', 'x040101', 'x040102', 'x040103'],
            ['x040200', 'x040201', 'x040202', 'x040203']]
        """
        return self.output_round_vars(self.nrounds - 1)

    def diff_input_vars(self):
        """
        Return a list of variables for the input difference of the Xoodoo permutation

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=3)
            >>> X = xoodoo.diff_input_vars()
            >>> [[X[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)] # doctest: +NORMALIZE_WHITESPACE
            [['X000000', 'X000001', 'X000002', 'X000003'],
            ['X000100', 'X000101', 'X000102', 'X000103'],
            ['X000200', 'X000201', 'X000202', 'X000203']]
        """
        return self.diff_input_round_vars(0)

    def diff_output_vars(self):
        """
        Return a list of variables for the output difference of the Xoodoo permutation

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=3)
            >>> X = xoodoo.diff_output_vars()
            >>> [[X[i][j].name for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)] # doctest: +NORMALIZE_WHITESPACE
            [['X030000', 'X030001', 'X030002', 'X030003'],
            ['X030100', 'X030101', 'X030102', 'X030103'],
            ['X030200', 'X030201', 'X030202', 'X030203']]
        """
        return self.diff_output_round_vars(self.nrounds - 1)

    def diff_round_constraints(self, r):
        """
        Return a list of constraints for the differential propagation for the `r`-th round of Xoodoo

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> constraints = xoodoo.diff_round_constraints(0)
            >>> for constraint in constraints:
            ...     xoodoo.solver.add(constraint)
            >>> xoodoo.solver.check()
            True
        """
        return self.diff_theta_constraints(r) + self.diff_rho_west_constraints(r) + self.diff_chi_constraints(r) +\
               self.diff_rho_east_constraints(r)

    def differential_constraints(self):
        """
        Return a list of constraints for the differential propagation of Xoodoo

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=3)
            >>> constraints = xoodoo.differential_constraints()
            >>> for constraint in constraints:
            ...     xoodoo.solver.add(constraint)
            >>> x = xoodoo.diff_input_round_vars(0)
            >>> xoodoo.solver.check()
            True
        """
        x = self.diff_input_vars()

        constraints = [
            (x[0][0] | x[0][1] | x[0][2] | x[0][3] |
             x[1][0] | x[1][1] | x[1][2] | x[1][3] |
             x[2][0] | x[2][1] | x[2][2] | x[2][3]) != 0
        ]
        constraints += sum([self.diff_round_constraints(r) for r in range(self.nrounds)], [])
        constraints += [self.total_weight_constraint()]

        return constraints

    def hamming_weight_constraint(self, x, weight):
        """
        Return a constraint represent the Hamming weight of `x`

        INPUT:

        - ``x`` -- word variable
        - ``weight`` -- the specified Hamming weight

        TESTS::

            >>> import random
            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> solver = xoodoo.solver
            >>> a = solver.bitvec('a', width=XoodooState.word_size)
            >>> w = random.randint(0, XoodooState.word_size)
            >>> solver.add(xoodoo.hamming_weight_constraint(a, w))
            >>> solver.check()
            True
            >>> result = solver.model(a.name)
            >>> bin(result)[2:].count('1') == w
            True
        """
        return sum([(x & (1 << i)) >> i for i in range(XoodooState.word_size)]) == weight

    def weight_vars(self, r):
        """
        Return a list of variables for the Hamming weight of column-wise Xoodoo state at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> W = xoodoo.weight_vars(0)
            >>> [w.name for w in W]
            ['w0000', 'w0001', 'w0002', 'w0003']
        """
        return self.aux_vars(r, Xoodoo._weight_var_name)

    def total_weight_var(self):
        """
        Return the variable representing the total Hamming weight of the differential trail

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=4)
            >>> xoodoo.total_weight_var().name == Xoodoo._total_weight_var_name
            True
        """
        return self.solver.bitvec(Xoodoo._total_weight_var_name, width=XoodooState.word_size)

    def total_weight_constraint(self):
        """
        Return a constraint representing the total weight for the differential trail

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> import random
            >>> xoodoo = Xoodoo(nrounds=2)
            >>> solver = xoodoo.solver
            >>> weight = random.randint(0, 256)
            >>> solver.add(xoodoo.total_weight_constraint())
            >>> solver.add(xoodoo.total_weight_var() == weight)
            >>> xoodoo.solver.check()
            True
            >>> weight_vars = sum([xoodoo.weight_vars(r) for r in range(xoodoo.nrounds)], [])
            >>> sum([xoodoo.solver.model(w.name) for w in weight_vars]) == weight
            True
        """
        w = sum([self.weight_vars(r) for r in range(self.nrounds)], [])
        W = self.total_weight_var()

        return sum(w) == W

    def differential_trail_constraints(self, w):
        """
        Return a list of constraints to find differential trail of trail weight `w`

        INPUT:

        -  ``w`` -- a positive integer
        """
        if w <= 0:
            raise ValueError("w must be a positive integer")

        constraints = self.differential_constraints()
        constraints += [self.total_weight_var() == w//2]

        return constraints

    def validity_input_vars(self, r):
        """
        Return a list of input variables to verify the validity of differential at round `r`

        INPUT:

        - ``r`` -- round number
        """
        x = self.input_chi_vars(r)
        a = self.diff_input_chi_vars(r)
        return [[x[i][j] + a[i][j] for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)]

    def validity_output_vars(self, r):
        """
        Return a list of output variables to verify the validity of differential at round `r`

        INPUT:

        - ``r`` -- round number
        """
        y = self.output_chi_vars(r)
        b = self.diff_output_chi_vars(r)
        return [[y[i][j] + b[i][j] for j in range(XoodooState.ncols)] for i in range(XoodooState.nrows)]

    def validity_round_constraints(self, r):
        """
        Return a list of constraints to verify the validity of a differential at round `r`

        INPUT:

        - ``r`` -- round number

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=3)
            >>> solver = xoodoo.solver
            >>> for constraint in xoodoo.validity_round_constraints(0):
            ...     solver.add(constraint)
            >>> solver.check()
            True
        """
        x = self.validity_input_vars(r)
        y = self.validity_output_vars(r)
        return [y[i][j] == x[i][j] ^ (~x[(i + 1) % XoodooState.nrows][j] & x[(i + 2) % XoodooState.nrows][j])
                for j in range(XoodooState.ncols) for i in range(XoodooState.nrows)]

    def validity_constraints(self):
        """
        Return a list of constraints to verify the validity of differential trail

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> xoodoo = Xoodoo(nrounds=2)
            >>> solver = xoodoo.solver
            >>> for constraint in xoodoo.validity_constraints():
            ...     solver.add(constraint)
            >>> solver.check()
            True
        """
        return sum([self.validity_round_constraints(r) for r in range(self.nrounds)], [])

    def has_differential_trail(self, w):
        """
        Return `True` if there exists a differential trail with trail weight `w`

        Note that the trail weight of a word is equal to twice of its Hamming weight

        INPUT:

        - ``w`` -- a positive integer

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> _2rounds_xoodoo_0 = Xoodoo(nrounds=2)
            >>> _2rounds_xoodoo_0.has_differential_trail(8)
            True
            >>> _2rounds_xoodoo_1 = Xoodoo(nrounds=2)
            >>> _2rounds_xoodoo_1.has_differential_trail(7)
            False
        """
        for constraint in self.differential_trail_constraints(w):
            self.solver.add(constraint)

        return self.solver.check()

    def has_valid_differential_trail(self, w):
        """
        Return `True` if there exists a valid differential trail with trail weight `w`

        INPUT:

        - ``w`` -- a positive integer

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> _2rounds_xoodoo_0 = Xoodoo(nrounds=2)
            >>> _2rounds_xoodoo_0.has_valid_differential_trail(8)
            True
            >>> _2rounds_xoodoo_1 = Xoodoo(nrounds=2)
            >>> _2rounds_xoodoo_1.has_valid_differential_trail(7)
            False
        """
        constraints = self.differential_trail_constraints(w) + self.validity_constraints() +\
                      self.permutation_constraints()

        for constraint in constraints:
            self.solver.add(constraint)

        return self.solver.check()

    def differential_trail(self, w):
        """
        Return a differential trail (if exists) with trail weight `w`

        INPUT:

        - ``w`` -- a positive integer

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> _2rounds_xoodoo_0 = Xoodoo(nrounds=2)
            >>> trail_0 = _2rounds_xoodoo_0.differential_trail(8)
            >>> _2rounds_xoodoo_1 = Xoodoo(nrounds=2)
            >>> trail_1 = _2rounds_xoodoo_1.differential_trail(7)
            Traceback (most recent call last):
            ...
            RuntimeError: no differential trail with trail weight 7
        """
        if not self.has_differential_trail(w):
            raise RuntimeError("no differential trail with trail weight %d" % w)

        return self._differential_trail_()

    def min_differential_trail(self):
        """
        Return a differential trail with minimum trail weight

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=2)
            >>> trail = X.min_differential_trail()
            >>> trail.weight()
            8
        """
        trail = None
        max_trail_weight = XoodooState.word_size * XoodooState.ncols
        for w in range(2, max_trail_weight, 2):
            other_instance = Xoodoo(nrounds=self.nrounds)
            try:
                trail = other_instance.differential_trail(w)
            except RuntimeError:
                continue
            break

        return trail

    def valid_differential_trail(self, w):
        """
        Return a valid differential trail (if exists) with trail weight `w`

        INPUT:

        - ``w`` -- a positive integer

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> _2rounds_xoodoo_0 = Xoodoo(nrounds=2)
            >>> trail_0 = _2rounds_xoodoo_0.valid_differential_trail(8)
            >>> _2rounds_xoodoo_1 = Xoodoo(nrounds=2)
            >>> trail_1 = _2rounds_xoodoo_1.valid_differential_trail(7)
            Traceback (most recent call last):
            ...
            RuntimeError: no valid differential trail with trail weight 7
        """
        if not self.has_valid_differential_trail(w):
            raise RuntimeError("no valid differential trail with trail weight %d" % w)

        return self._differential_trail_()

    def min_valid_differential_trail(self):
        """
        Return a valid differential trail with minimum trail weight

        TESTS::

            >>> from xoodoo import Xoodoo
            >>> X = Xoodoo(nrounds=2)
            >>> trail = X.min_valid_differential_trail()
            >>> trail.weight()
            8
        """
        trail = None
        max_trail_weight = XoodooState.word_size * XoodooState.ncols
        for w in range(2, max_trail_weight, 2):
            other_instance = Xoodoo(nrounds=self.nrounds)
            try:
                trail = other_instance.valid_differential_trail(w)
            except RuntimeError:
                continue
            break

        return trail

    def valid_input(self, w):
        """
        Return an input that satisfy the differential trail (if exists) with trail weight `w`

        INPUT:

        - ``w`` -- a positive integer

        TESTS::

            >>> from xoodoo import Xoodoo, XoodooState
            >>> _2rounds_xoodoo_0 = Xoodoo(nrounds=2)
            >>> x = _2rounds_xoodoo_0.valid_input(8)
            >>> isinstance(x, XoodooState)
            True
        """
        if not self.has_valid_differential_trail(w):
            raise RuntimeError("no valid differential trail with trail weight %d" % w)

        x = self.input_variables()
        model = self.solver.model()
        nrows = XoodooState.nrows
        ncols = XoodooState.ncols

        return XoodooState([model[x[i][j].name] for i in range(nrows) for j in range(ncols)])

    def _differential_trail_(self):
        """
        Return XoodooTrail object from `self.solver.model()`
        """
        model = self.solver.model()
        nrows = XoodooState.nrows
        ncols = XoodooState.ncols

        trail = []
        for r in range(self.nrounds):
            a = self.diff_input_round_vars(r)
            trail.append([model[a[i][j].name] for i in range(nrows) for j in range(ncols)])

            b = self.diff_input_rho_west_vars(r)
            trail.append([model[b[i][j].name] for i in range(nrows) for j in range(ncols)])

            c = self.diff_input_chi_vars(r)
            trail.append([model[c[i][j].name] for i in range(nrows) for j in range(ncols)])

            d = self.diff_input_rho_east_vars(r)
            trail.append([model[d[i][j].name] for i in range(nrows) for j in range(ncols)])

        y = self.diff_output_vars()
        trail.append([model[y[i][j].name] for i in range(nrows) for j in range(ncols)])

        return XoodooTrail(trail)

    def __repr__(self):
        """
        Return a string representation of Xoodoo object

        EXAMPLES::

            >>> from xoodoo import Xoodoo
            >>> Xoodoo(nrounds=1)
            1-round of Xoodoo
            >>> Xoodoo(nrounds=2)
            2-rounds of Xoodoo
        """
        round_str = "round"
        if self.nrounds > 1:
            round_str += 's'

        return "%d-%s of Xoodoo" % (self.nrounds, round_str)
