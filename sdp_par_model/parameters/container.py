"""
Handles collections of telescope parameters. This module contains
all the necessary plumbing to make the parameter definitions work.
"""

from sympy import symbols, Symbol, Expr, Lambda, Mul, Add, Sum
import warnings
import string

def is_value(self, v):
    if isinstance(v, int) or isinstance(v, float) or \
       isinstance(v, str) or isinstance(v, list):
        return True

def is_expr(self, e):
    return isinstance(e, Expr) or isinstance(e, BLDep)

class ParameterContainer(object):
    """Stores calculated telescope parameters.

    All fields set on objects are either inputs or outputs of
    telescope parameter calculations. We expect all fields to have one
    of the following types:

    * Simple value types such as integer, float, string or list. These
      are assumed to be constants or calculated values.

    * Sympy expressions for not entirely evaluated values. Appear if
      parameters were left open, such as if the number of facets was
      not decided yet, or we are evaluating the model symbolically.

    * Baseline-dependent expressions (BLDep, see below). Expressions
      that have a different value depending on the considered
      baseline.
    """
    def __init__(self):
        pass

    def __str__(self):
        s = "Parameter Container Object with the following fields:"
        fields = self.__dict__
        for k in fields.keys():
            key_string = str(k)
            value_string = str(fields[k])
            if len(value_string) > 40:
                value_string = value_string[:40] + "... (truncated)"
            s += "\n%s\t\t= %s" % (key_string, value_string)
        return s

    def set_param(self, param_name, value, prevent_overwrite=True, require_overwrite=False):
        """
        Provides a method for setting a parameter. By default first checks that the value has not already been defined.
        Useful for preventing situations where values may inadvertently be overwritten.
        @param param_name: The name of the parameter/field that needs to be assigned - provided as text
        @param value: the value. Need not be text.
        @param prevent_overwrite: Disallows this value to be overwritten once defined. Default = True.
        @return: Nothing
        """
        assert isinstance(param_name, str)
        if prevent_overwrite:
            if require_overwrite:
                raise AssertionError("Cannot simultaneously require and prevent overwrite of parameter '%s'" % param_name)

            if hasattr(self, param_name):
                if eval('self.%s == value' % param_name):
                    # warnings.warn('Inefficiency Warning: reassigning already-defined parameter "%s" with an identical value.' % param_name)
                    pass
                else:
                    try:
                        assert eval('self.%s == None' % param_name)
                    except AssertionError:
                        raise AssertionError("The parameter %s has already been defined and may not be overwritten." % param_name)

        elif require_overwrite and (not hasattr(self, param_name)):
            raise AssertionError("Parameter '%s' is undefined and therefore cannot be assigned" % param_name)

        exec('self.%s = value' % param_name)  # Write the value

    def get_param(self, param_name):
        """
        Provides a method for reading a parameter that won't necessarily crash the program if the parameter has not yet
        been defined. A nice-to-have robustness-enhancing feature but not really necessary to get a working implementation.
        @param param_name: The name of the parameter/field that needs to be read - provided as text
        @return: The parameter value.
        """
        assert isinstance(param_name, str)
        return_value = None
        try:
            assert hasattr(self, param_name)
            exec('return_value = self.%s' % param_name)  # Write the value to the return_value variable
        except AssertionError:
            warnings.warn("Parameter %s hasn't been defined (returning 'None')." % param_name)

        return return_value

    def make_symbol_name(self, name):
        """Make names used in our code into something more suitable to be used
        as a Latex symbol. This is a quick-n-dirty heuristic based on
        what the names used in equations.py tend to look like.
        """

        if name.startswith("wl"):
            return 'lambda' + name[2:]
        if name.startswith("freq_"):
            return 'f_' + name[5:]
        if name.startswith("Delta"):
            return 'Delta_' + name[5:]
        if name.startswith("Theta_"):
            return 'Theta_' + name[6:].replace('_', ',')
        if name.startswith("Omega_"):
            return name
        if name[0].isupper():
            i0 = 2 if name[1] == '_' else 1
            return name[0] + "_" + name[i0:].replace('_', ',')
        return name

    def symbolify(self):

        # Replace all values and expressions with symbols
        for name, v in self.__dict__.items():
            sym = Symbol(self.make_symbol_name(name), real=True, positive=True)
            if isinstance(v, int) or isinstance(v, float) or isinstance(v, Expr):
                self.__dict__[name] = sym
            elif isinstance(v, BLDep):
                # SymPy cannot pass parameters by dictionary, so make a list instead
                self.__dict__[name] = BLDep(v.pars, sym(*v.pars.values()))

        # For products too
        for product, rates in self.products:
            for rname in rates:
                rates[rname] = Symbol(self.make_symbol_name(rname + "_" + product))

    def get_products(self, expression='Rflop', scale=1):
        results = {}
        for product, exprs in self.products.items():
            if expression in exprs:
                results[product] = exprs[expression] / scale
        return results

    def _sum_bl_bins(self, bldep, bmax_bins=None, bcount_bins=None):
        """
        Converts a possibly baseline-dependent terms (e.g. constructed
        using "BLDep" or "blsum") into a formula by summing over
        baselines.

        If "bmax_bins" and/or "bcount_bins" parameters are not set we
        default on "Nbl_full", "frac_bins" and "Bmax_bins" fields to
        be set. If "bmax_bins" is a symbol, we will generate a
        symbolic sum term utilising "bmax_bins" as a function.
        """

        # Actually baseline-dependent?
        if not isinstance(bldep, BLDep):
            return self.Nbl * bldep
        expr = bldep.term

        # Bin defaults
        if bmax_bins is None:
            bmax_bins = self.Bmax_bins
        if bcount_bins is None:
            bcount_bins = [ frac * self.Nbl_full for frac in self.frac_bins ]

        # Small bit of ad-hoc formula optimisation: Exploit
        # independent factors. Makes for smaller terms, which is good
        # both for Sympy as well as for output.
        pars = bldep.pars.values()
        def independent(e):
            return not any([s in e.free_symbols for s in pars])
        if independent(expr):
            return self.Nbl * bldep(b=None, bcount=1)
        if isinstance(expr, Mul):
            indepFactors = list(filter(independent, expr.as_ordered_factors()))
            if len(indepFactors) > 0:
                def not_indep(e): return not independent(e)
                restFactors = filter(not_indep, expr.as_ordered_factors())
                return Mul(*indepFactors) * self._sum_bl_bins(BLDep(bldep.pars, Mul(*restFactors)),
                                                              bmax_bins=bmax_bins, bcount_bins=bcount_bins)

        # Symbolic? Generate actual symbolic sum expression
        if isinstance(bmax_bins, Symbol):
            b = Symbol('b')
            return Sum(bldep(b=bmax_bins(b), bcount=1), (b, 1, self.Nbl))

        # Otherwise generate sum term manually that approximates the
        # full sum using baseline bins
        results = [ bldep(b=bmax_val, bcount=bcount_val)
                    for bmax_val, bcount_val in zip(bmax_bins, bcount_bins) ]
        return Add(*results, evaluate=False)

    def set_product(self, product, T=None, N=1, bmax_bins=None, bcount_bins=None, **args):
        """Sets product properties using a task abstraction. Each property is
        expressed as a sum over baselines.

        @param product: Product to set.
        @param T: Observation time covered by this task. Default is the
          entire observation (Tobs). Can be baseline-dependent.
        @param N: Task parallelism / rate multiplier. The number of
           tasks that work on the data in parallel. Can be
           baseline-dependent.
        @param bmax_bins: Maximum lengths of baseline bins to use
        @param bcount_bins: Size of baseline bins to use
        @param args: Task properties as rates. Will be multiplied by
           N.  If it is baseline-dependent, it will be summed over all
           baselines to yield the final rate.
        """

        # Collect properties
        if T is None: T = self.Tobs
        props = { "N": N, "T": T }
        for k, expr in args.items():

            # Multiply out multiplicator. If either of them is
            # baseline-dependent, this will generate a new
            # baseline-dependent term (see BLDep)
            total = N * expr

            # Baseline-dependent? Generate a sum term, otherwise just say as-is
            if isinstance(total, BLDep):
                props[k] = self._sum_bl_bins(total, bmax_bins, bcount_bins)
                props[k+"_task"] = expr
            else:
                props[k] = total

        # Update
        if not product in self.products:
            self.products[product] = {}
        self.products[product].update(props)

class BLDep(object):
    """A baseline-dependent sympy expression.

    Named baseline properties can be used as symbols in the sympy
    expression. Typical choices would be 'b' for the baseline length
    or 'bcount' for the baseline count.

    Note that this mostly replicates functionality of numpy's own
    Lambda expression. The main difference here are that we assign
    semantics to the term and parameters (e.g. baseline
    properties). Furthermore, we also lift some arithmetic operations
    such that they can also work on baseline-dependent terms.
    """

    def __init__(self, pars, term):
        """
        Creates baseline-dependent term.
        :param pars: List of baseline-dependent parameters as
          dictionary of Symbols. If only a single symbol is given, it
          will stand for baseline length.
        :param term: Dependent term, in which "pars" symbols can appear
          free and will be substituted later.
        """
        self.term = term
        # Collect formal parameters. We default to parameter name 'b'
        if not isinstance(pars, dict):
            self.pars = { 'b': pars }
        else:
            self.pars = pars
        non_symbols = [p for p in self.pars.values() if not isinstance(p, Symbol)]
        assert len(non_symbols) == 0, "Formal parameters %s are not a symbol!" % non_symbols

    def __call__(self, vals=None, **kwargs):
        """
        Evaluates baseline-dependent term. If only a single parameter is
        given, it is assumed to be baseline length. Additional parameters
        can be passed as dictionary or keyword arguments. The following is
        equivalent:

           bldep(x)
           bldep({'b': x})
           bldep(b=x)
        """
        if not isinstance(self.term, Expr):
            return self.term
        # Check that all parameters were passed
        if vals is None:
            vals = {}
        elif not isinstance(vals, dict):
            vals = { 'b': vals }
        vals.update(kwargs)
        assert set(self.pars.keys()).issubset(vals.keys()), \
            "Parameter %s not passed to baseline-dependent term %s! %s" % (
                set(self.pars.keys()).difference(vals.keys()), self.term, vals)
        # Do substitutions
        to_substitute = [(p, vals[p]) for p in self.pars.keys()]
        return self.term.subs(to_substitute)

    def __mul__(self, other):
        # Other term not baseline-dependent?
        if not isinstance(other, BLDep):
            return BLDep(self.pars, self.term * other)
        if not isinstance(self.term, Expr):
            return other * self.term
        # Determine required renamings
        renamings = {
            pold : other.pars[name]
            for name, pold in self.pars.items()
            if name in other.pars
        }
        # Adapt new parameters & term
        newpars = self.pars.copy()
        newpars.update(other.pars)
        newterm = self.term.subs(renamings.items())
        return BLDep(newpars, newterm * other.term)
    def __rmul__(self, other):
        return self.__mul__(other)

    def subs(self, *args, **kwargs):
        if not isinstance(self.term, Expr):
            return self
        return BLDep(self.pars, self.term.subs(*args, **kwargs))

    @property
    def free_symbols(self):
        return Lambda(list(self.pars.values()), self.term).free_symbols
    def atoms(self, typ):
        return Lambda(list(self.pars.values()), self.term).atoms(typ)

def blsum(b, expr):
    """
    A baseline sum of an expression.

    Returns a BLDep object of the expression multiplied with the baseline count.
    """
    bcount = Symbol('bcount')
    return BLDep({'b': b, 'bcount': bcount}, bcount * expr)