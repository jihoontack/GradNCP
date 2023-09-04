def setup(mode, P):
    if mode in ['maml']:
        from evals.gradient_based.maml import test_model as test_func
    elif mode in ['maml_bootstrap_param']:
        from evals.gradient_based.maml_scale import test_model as test_func
    elif mode in ['maml_full_evaluate', 'maml_full_evaluate_gradscale']:
        from evals.gradient_based.maml_full_evaluate import test_model as test_func
    else:
        print(f'Warning: current running option, i.e., {mode}, needs evaluation code')
        from evals.gradient_based.maml import test_model as test_func

    return test_func
