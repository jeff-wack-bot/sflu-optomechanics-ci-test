"""
"""
import os
from gwinc.struct import Struct


def optickle_model(basename, params=None):
    """
    Decorator to run optickle models once and save the results in a cached_models
    directory for future use. Intended to be used as a pytest fixture like

    @pytest.fixture
    @optickle_model('optname', params=params)
    def optFP():
        eng = start_matlab_engine()
        import qlance.optickle as qopt
        opt = qopt.Optickle(eng, *args, **kwargs)
        # code to build optickle model
        # code to run the optickle model
        return opt

    The basename is the file name that the results are exported to. The optional
    params kwarg is a gwinc Struct containing parameters of the model. If given
    that struct will also be cached. The model will be (re)computed if no hdf5
    cache exists or if pars is given and different from the cached parameter struct.
    The evaluated optickle model will just be loaded and returned if the cached
    struct is the same as the pars struct.

    The exported optickle model will be saved in
    cached_models/optname_model.h5
    and the cached parameters struct will be saved in
    cached_models/optname_params.yaml
    """
    def decorator(func):
        test_path = os.path.split(__file__)[0]
        mpath = os.path.join(test_path, 'cached_models')
        os.makedirs(mpath, exist_ok=True)
        mname = os.path.join(mpath, basename + '_model.h5')
        pname = os.path.join(mpath, basename + '_params.yaml')

        def saved_model(*args, **kwargs):
            print('loading')
            from qlance.plant import OpticklePlant
            opt = OpticklePlant()
            opt.load(mname)
            return opt

        def new_model(*args, **kwargs):
            print('executing')
            opt = func(*args, **kwargs)
            print('saving')
            opt.save(mname)
            if params:
                print('saving parameters')
                params.to_yaml(pname)
            return opt

        if os.path.exists(mname):
            if params:
                try:
                    old_params = Struct.from_file(pname)
                    if old_params == params:
                        return saved_model
                    else:
                        return new_model
                except FileNotFoundError:
                    return new_model
            else:
                return saved_model
        else:
            return new_model

    return decorator


def start_matlab_engine():
    """
    Start the matlab engine and add the Optickle path to that engine.
    The initialized engine is returned.
    """
    from time import time
    import matlab.engine
    import qlance.optickle as qopt

    print('starting matlab engine')
    ts = time()
    eng = matlab.engine.start_matlab()
    te = time()
    print('engine started in {:0.2f} s'.format(te - ts))
    qopt.addOpticklePath(eng)
    return eng
