import streamlit as st
import joblib
import pandas as pd
from joblib import load
from tempfile import TemporaryDirectory
import xgboost
from pymatgen.core.composition import Composition
from ase.io import read
import os
import matplotlib.pyplot as plt
from matminer.featurizers.composition import (
    TMetalFraction, Stoichiometry, BandCenter, ElementProperty, Meredig
)
import shap

print("XGBoost version:", xgboost.__version__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def data_process(uploaded_files, model_path=os.path.join(BASE_DIR, "xgb_model.joblib")):
    """
    从上传的CIF文件生成模型输入特征
    使用SHAP可视化
    """
    featurizers = [
        BandCenter(impute_nan=True),
        Stoichiometry(p_list=[0], num_atoms=True),
        TMetalFraction(),
        ElementProperty.from_preset(preset_name="magpie", impute_nan=True),
        Meredig(impute_nan=True)
    ]

    model = load(model_path)
    template_df = pd.read_csv(
        os.path.join(BASE_DIR, "whole_featbase.csv"), 
        nrows=1
    )
    feature_columns = template_df.columns[3:].tolist()

    # 处理上传的CIF文件
    features = []
    with TemporaryDirectory() as tmpdir:
        for uploaded_file in uploaded_files:
            file_path = f"{tmpdir}/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            sample_features = []
            composition = Composition(read(file_path).get_chemical_formula())
            
            for featurizer in featurizers:
                sample_features.extend(featurizer.featurize(composition))
            temp_series = pd.Series(sample_features)
            aligned_features = temp_series.reindex(range(len(feature_columns)), fill_value=0).values
            features.append(aligned_features)

    # 生成最终DataFrame
    df_features = pd.DataFrame(features, columns=feature_columns)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_features)
    
    plt.figure()
    shap.summary_plot(shap_values, df_features, plot_type="bar", show=False)
    plt.tight_layout()
    
    return df_features, plt.gcf()

st.title("晶体结构预测高温超导材料")
st.write("""
    本应用程序使用XGBoost模型对输入数据进行预测。
""")

@st.cache_data
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "xgb_model.joblib"))

model = load_model()

uploaded_file = st.file_uploader("上传测试数据（cif格式）", accept_multiple_files=True)

if uploaded_file:
    features, shap_plot = data_process(uploaded_file)
    st.write("生成特征：", features)
    st.pyplot(shap_plot)

if st.button("运行预测"):
    predictions = model.predict(features)
    st.line_chart(predictions)
