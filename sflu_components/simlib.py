"""
Functions for checking with Optickle and Finesse models (using Qlance)
"""
import os
from gwinc.struct import Struct
import inspect
from hashlib import sha1


def optickle_model(basename, params=None):
    """
    Decorator to run optickle models once and save the results in a cached_models
    directory for future use. Models are rerun if the function generating a model
    changes or parameters used by that function change.

    Note that the models must be run: models without run already called will not work.

    Can be used as a pytest fixture like

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
    params kwarg is a gwinc Struct containing parameters of the model. If given,
    that struct will also be cached. The function code is also hashed and saved.

    The model will be (re)computed if any of the following are true:
      * no model hdf5 or text hash files exist
      * the code generating the function changes (as determined by the funcion hash)
      * the params struct, if given, is different from the cached parameter struct
    Otherwise the evaluated optickle model will be loaded and returned instead.

    The exported optickle model is saved in
      cached_models/basename_optickle_model.h5
    The cached parameters struct is saved in
      cached_models/basename_optickle_params.yaml
    The function hash is saved in
      cached_models/basename_optickle_hash.txt
    """
    def decorator(func):
        test_path = os.path.split(__file__)[0]
        mpath = os.path.join(test_path, 'cached_models')
        os.makedirs(mpath, exist_ok=True)
        mname = os.path.join(mpath, basename + '_optickle_model.h5')
        pname = os.path.join(mpath, basename + '_optickle_params.yaml')
        hname = os.path.join(mpath, basename + '_optickle_hash.txt')

        # native python hash() doesn't reliably work
        func_code = inspect.getsource(func)
        hash_string = sha1(bytes(func_code, 'utf-8')).hexdigest()
        print('\nnew hash', hash_string)

        def saved_model(*args, **kwargs):
            print('\nloading')
            from qlance.plant import OpticklePlant
            opt = OpticklePlant()
            opt.load(mname)
            return opt

        def new_model(*args, **kwargs):
            print('\nexecuting')
            opt = func(*args, **kwargs)
            print('saving')
            opt.save(mname)
            with open(hname, 'w') as fh:
                fh.write(hash_string)
            if params:
                print('saving parameters')
                params.to_yaml(pname)
            return opt

        if os.path.exists(mname) and os.path.exists(hname):
            with open(hname, 'r') as fh:
                old_hash = fh.read()
                print('old hash', old_hash)
            if old_hash != hash_string:
                return new_model

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

################################################################################
# Finesse
################################################################################


def finesse_model(basename, params=None):
    """
    Decorator to run finesse models once and save the results in a cached_models
    directory for future use. Models are rerun if the function generating a model
    changes or parameters used by that function change.

    Note that the models must be run: models without run already called will not work.

    Can be used as a pytest fixture like

    @pytest.fixture
    @finesse_model('optname', params=params)
    def katFP():
        from pykat import finesse
        import qlance.finesse as fin
        kat = finesse.kat()
        # code to build finesse model
        katFR = qfin.KatFR(kat)
        # code to run the finesse model
        return katFR

    The basename is the file name that the results are exported to. The optional
    params kwarg is a gwinc Struct containing parameters of the model. If given,
    that struct will also be cached. The function code is also hashed and saved.

    The model will be (re)computed if any of the following are true:
      * no model hdf5 or text hash files exist
      * the code generating the function changes (as determined by the funcion hash)
      * the params struct, if given, is different from the cached parameter struct
    Otherwise the evaluated finesse model will be loaded and returned instead.

    The exported finesse model is saved in
      cached_models/basename_finesse_model.h5
    The cached parameters struct is saved in
      cached_models/basename_finesse_params.yaml
    The function hash is saved in
      cached_models/basename_finesse_hash.txt
    """
    def decorator(func):
        test_path = os.path.split(__file__)[0]
        mpath = os.path.join(test_path, 'cached_models')
        os.makedirs(mpath, exist_ok=True)
        mname = os.path.join(mpath, basename + '_finesse_model.h5')
        pname = os.path.join(mpath, basename + '_finesse_params.yaml')
        hname = os.path.join(mpath, basename + '_finesse_hash.txt')

        # native python hash() doesn't reliably work
        func_code = inspect.getsource(func)
        hash_string = sha1(bytes(func_code, 'utf-8')).hexdigest()
        print('\nnew hash', hash_string)

        def saved_model(*args, **kwargs):
            print('\nloading')
            from qlance.plant import FinessePlant
            katFR = FinessePlant()
            katFR.load(mname)
            return katFR

        def new_model(*args, **kwargs):
            print('\nexecuting')
            katFR = func(*args, **kwargs)
            print('saving')
            katFR.save(mname)
            with open(hname, 'w') as fh:
                fh.write(hash_string)
            if params:
                print('saving parameters')
                params.to_yaml(pname)
            return katFR

        if os.path.exists(mname) and os.path.exists(hname):
            with open(hname, 'r') as fh:
                old_hash = fh.read()
                print('old hash', old_hash)
            if old_hash != hash_string:
                return new_model

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
