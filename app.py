from clients.googlefit_client import GoogleFitClient

# ...

st.header("Google Fit Connection")

col1, col2 = st.columns(2)
with col1:
    days_back = st.number_input("Days back", min_value=1, max_value=90, value=14, step=1)
with col2:
    if st.button("Test Google Fit nutrition"):
        try:
            gf = GoogleFitClient()
            daily_macros = gf.get_daily_macros(days_back=days_back)

            if daily_macros.empty:
                st.info("Google Fit returned no nutrition entries for the selected period.")
            else:
                st.success("Google Fit connection OK - daily calories and macros aggregated")
                st.dataframe(daily_macros)
        except Exception as e:
            st.error(str(e))


if st.button("Show raw Google Fit dataset"):
    gf = GoogleFitClient()
    st.json(gf.get_nutrition_dataset())