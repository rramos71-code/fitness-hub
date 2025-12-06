import streamlit as st

def get_secret(key: str):
    secrets = st.secrets.get("fitness_hub", {})
    return secrets.get(key)
