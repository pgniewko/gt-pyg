# Third party
import weightwatcher as ww


def get_ww_statistic(model, statistic="alpha"):
    assert statistic in [
        "log_norm",
        "alpha",
        "alpha_weighted",
        "log_alpha_norm",
        "log_spectral_norm",
        "stable_rank",
    ]
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(plot=False)
    summary = watcher.get_summary(details)
    return summary[statistic]


def ww_active_layer_idx(model):
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(plot=False)
    return details["layer_id"].to_list()


def ww_active_layer_alpha(model):
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(plot=False)
    return details["alpha"].to_list()


def ww_active_layers_data(model):
    indices = ww_active_layer_idx(model)
    alphas = ww_active_layer_alpha(model)

    names = []
    layer_alpha = []
    for idx, module in enumerate(model.named_modules()):
        if idx in indices:
            names.append(module[0])
            layer_alpha.append(alphas[indices.index(idx)])

    assert len(names) == len(indices)
    return (names, layer_alpha)


def prepare_params(model, base_lr):
    """
    Usage:
      lr = 0.001
      params = prepare_params(model, lr)
      optimizer = torch.optim.AdamW(params)
    """
    params = []
    for name, p in model.named_parameters():
        params.append({"params": p, "lr": base_lr, "name": name, "base_lr": base_lr})

    return params


def get_correction_factor(model, opt):
    (ln, avals) = ww_active_layers_data(model)

    corrections = []
    for pg in opt.param_groups:
        name = ".".join(pg["name"].split(".")[:-1])
        if name in ln:
            alpha = avals[ln.index(name)]
            if alpha > 2.0 and alpha < 3.0:
                corr = 0.0
            if alpha > 3.0 and alpha < 4.0:
                corr = 0.25
            if alpha > 4.0 and alpha < 5.0:
                corr = 0.5
            if alpha > 5.0 and alpha < 6.0:
                corr = 0.75
            if alpha < 2.0:
                corr = 1.0
            if alpha > 6.0:
                corr = 1.0 + (alpha - 6.0)
            corrections.append(corr)

        else:
            corrections.append(1.0)

    return corrections
