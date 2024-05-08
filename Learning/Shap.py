# %%
# os.getcwd() #os.chdir to change directory
# %%
import os
import shap as shp
import xgboost

shp.initjs() #Initialize JavaScript visualization code
# %%
# train an XGBoost model
X, y = shp.datasets.california()
model = xgboost.XGBRegressor().fit(X, y)
# %% explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shp.Explainer(model)
shap_values = explainer(X)
# %% visualize the first prediction's explanation
shp.plots.waterfall(shap_values[0])
shp.plots.force(shap_values[0]) #not working
# %% static plot
shp.plots.force(shap_values[0], matplotlib=True)
# %%
shp.plots.force(shap_values[:500])
# %%
