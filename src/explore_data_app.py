from glob import glob

import pandas as pd
import streamlit as st

DATA_ROOT = "../debug_data/phoenix-2014-multisigner/"


st.title("Explore Data")

st.header("CSV Info")

phoenix_df = pd.read_csv(DATA_ROOT + "annotations/manual/dev.corpus.csv", delimiter="|")
phoenix_df = phoenix_df.head(5)

st.write(phoenix_df)

st.header("Examples")

for row in phoenix_df[["folder", "annotation"]].values:
    folder, annotation = row
    st.write(folder)
    st.write(annotation)
    frames = glob(DATA_ROOT + "features/fullFrame-210x260px/" + folder)
    st.image(frames[:12], width=100)
