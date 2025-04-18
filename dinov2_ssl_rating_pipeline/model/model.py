from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor


def get_model(cfg):
    model_type = cfg.model.type.lower()

    if model_type == "ridge":
        return Ridge(alpha=cfg.model.alpha)

    elif model_type == "lasso":
        return Lasso(alpha=cfg.model.alpha)

    elif model_type == "svr":
        return SVR(C=cfg.model.svr_c, kernel=cfg.model.svr_kernel)

    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            random_state=cfg.model.random_state
        )

    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=cfg.model.n_estimators,
            learning_rate=cfg.model.learning_rate,
            max_depth=cfg.model.max_depth,
            random_state=cfg.model.random_state
        )

    elif model_type == "lightgbm":
        return LGBMRegressor(
            n_estimators=cfg.model.n_estimators,
            learning_rate=cfg.model.learning_rate,
            max_depth=cfg.model.max_depth,
            random_state=cfg.model.random_state
        )

    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
