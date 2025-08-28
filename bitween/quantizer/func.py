def _set_module(model, submodule_key, module):
    """
    Helper function to replace a module within a model hierarchy.
    
    Args:
        model (nn.Module): The main model.
        submodule_key (str): The dot-separated key to the submodule (e.g., "layer1.0.conv1").
        module (nn.Module): The new module to replace the old one.
    """
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)