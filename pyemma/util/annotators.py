
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
doc_inherit decorator

Usage:

class Foo(object):
    def foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @doc_inherit
    def foo(self):
        pass

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
"""
from __future__ import absolute_import
from functools import wraps
import warnings
from six import PY2
from decorator import decorator, decorate
from inspect import stack

__all__ = ['alias',
           'aliased',
           'deprecated',
           'doc_inherit',
           'shortcut',
           ]


class DocInherit(object):

    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit

class alias(object):
    """
    Alias class that can be used as a decorator for making methods callable
    through other names (or "aliases").
    Note: This decorator must be used inside an @aliased -decorated class.
    For example, if you want to make the method shout() be also callable as
    yell() and scream(), you can use alias like this:

        @alias('yell', 'scream')
        def shout(message):
            # ....
    """

    def __init__(self, *aliases):
        """Constructor."""
        self.aliases = set(aliases)

    def __call__(self, f):
        """
        Method call wrapper. As this decorator has arguments, this method will
        only be called once as a part of the decoration process, receiving only
        one argument: the decorated function ('f'). As a result of this kind of
        decorator, this method must return the callable that will wrap the
        decorated function.
        """
        f._aliases = self.aliases
        return f


def aliased(aliased_class):
    """
    Decorator function that *must* be used in combination with @alias
    decorator. This class will make the magic happen!
    @aliased classes will have their aliased method (via @alias) actually
    aliased.
    This method simply iterates over the member attributes of 'aliased_class'
    seeking for those which have an '_aliases' attribute and then defines new
    members in the class using those aliases as mere pointer functions to the
    original ones.

    Usage:
        @aliased
        class MyClass(object):
            @alias('coolMethod', 'myKinkyMethod')
            def boring_method(self):
                # ...

        i = MyClass()
        i.coolMethod() # equivalent to i.myKinkyMethod() and i.boring_method()
    """
    original_methods = aliased_class.__dict__.copy()
    for name, method in original_methods.items():
        if hasattr(method, '_aliases'):
            # Add the aliases for 'method', but don't override any
            # previously-defined attribute of 'aliased_class'
            for alias in method._aliases - set(original_methods):
                setattr(aliased_class, alias, method)
    return aliased_class


def shortcut(*names):
    """Add an shortcut (alias) to a decorated function, but not to class methods!
    
    use aliased/alias decorators for class members!

    Calling the shortcut (alias) will call the decorated function. The shortcut name will be appended
    to the module's __all__ variable and the shortcut function will inherit the function's docstring

    Examples
    --------
    In some module you have defined a function
    >>> @shortcut('is_tmatrix') # doctest: +SKIP
    >>> def is_transition_matrix(args): # doctest: +SKIP
    ...     pass # doctest: +SKIP
    Now you are able to call the function under its short name
    >>> is_tmatrix(args) # doctest: +SKIP

    """
    # TODO: this does not work (is not tested with class member functions)
    # it is not possible to reliably determine if a function is a member function, until it is bound
    def wrap(f):
        # TODO: this is wrong for class member shortcuts
        globals_ = f.__globals__ if PY2 else f.__globals__
        for name in names:
            globals_[name] = f
            if '__all__' in globals_ and name not in globals_['__all__']:
                globals_['__all__'].append(name)
        return f
    return wrap

def deprecated(*optional_message):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Parameters
    ----------
    *optional_message : str
        an optional user level hint which should indicate which feature to use otherwise.

    """
    def _deprecated(func, *args, **kw):
        caller_frame = stack()[1]
        filename = caller_frame[0].f_globals.get('__file__', None)
        lineno = func.__code__.co_firstlineno + 1

        user_msg = "Call to deprecated function %s. Called from %s line %i. %s" \
                   % (func.__name__, filename, lineno, msg)

        warnings.warn_explicit(
            user_msg,
            category=DeprecationWarning,
            filename=func.__code__.co_filename,
            lineno=func.__code__.co_firstlineno + 1
        )
        return func(*args, **kw)
    if len(optional_message) == 1 and callable(optional_message[0]):
        # this is the function itself, decorate!
        msg = ""
        return decorate(optional_message[0], _deprecated)
    else:
        # actually got a message (or empty parenthesis)
        msg = optional_message[0] if len(optional_message) > 0 else ""
        return decorator(_deprecated)

@decorator
def estimation_required(func, *args, **kw):
    """
    Decorator checking the self._estimated flag in an Estimator instance, raising a value error if the decorated
    function is called before estimator.estimate() has been called.

    If mixed with a property-annotation, this annotation needs to come first in the chain of function calls, i.e.,

    @property
    @estimation_required
    def func(self):
        ....
    """
    self = args[0] if len(args) > 0 else None
    if self and hasattr(self, '_estimated') and not self._estimated:
        raise ValueError("Tried calling %s on %s which requires the estimator to be estimated."
                         % (func.__name__, self.__class__.__name__))
    return func(*args, **kw)