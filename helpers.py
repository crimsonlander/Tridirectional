import tensorflow as tf
from collections import defaultdict


def conditional_reset(tensor, shape, cond):
    return tf.cond(cond, lambda: tf.zeros(shape), lambda: tensor)


class NameCreator(object):
    class_counter = defaultdict(int)

    @staticmethod
    def name_it(obj, name=None):
        if name:
            return name
        class_name = type(obj).__name__
        num = NameCreator.class_counter[class_name]
        NameCreator.class_counter[class_name] += 1
        name = "%s_%d" % (class_name, num)
        return name


def function_with_name_scope(method):
    def wrapper(self, *args, **kwargs):
        with tf.variable_scope(self.name):
            return method(self, *args, **kwargs)
    return wrapper


class_methods_with_scope = set()


def class_with_name_scope(cls):
    """
    Decorate every regular function of class with function_with_name_scope. Check if function is already decorated.
    Regular functions are all callable attributes of class which names don't start with '__'.
    """
    attribs = dir(cls)
    for attr in attribs:
        item = getattr(cls, attr)
        key = (cls.__name__, attr)
        if attr[:2] != '__' and hasattr(item, '__call__') and key not in class_methods_with_scope:
            setattr(cls, attr, function_with_name_scope(item))
            class_methods_with_scope.add(key)
    return cls