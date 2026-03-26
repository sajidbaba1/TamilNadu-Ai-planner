import streamlit as st
import os
import pandas as pd
from engine.engine_api import generate_plan
from engine.engine import NBC_MIN_AREA
from renderer.renderer import render

DISTRICTS = ['Ariyalur', 'Chengalpattu', 'Chennai', 'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul', 'Erode', 'Kallakurichi', 'Kancheepuram', 'Kanyakumari', 'Karur', 'Krishnagiri', 'Madurai', 'Mayiladuthurai', 'Nagapattinam', 'Namakkal', 'Nilgiris', 'Perambalur', 'Pudukkottai', 'Ramanathapuram', 'Ranipet', 'Salem', 'Sivaganga', 'Tenkasi', 'Thanjavur', 'Theni', 'Thiruvallur', 'Thiruvarur', 'Thoothukudi', 'Tiruchirappalli', 'Tirunelveli', 'Tirupathur', 'Tiruppur', 'Tiruvanamalai', 'Vellore', 'Villupuram', 'Virudhunagar']

st.set_page_config(layout="wide", page_title="TN AI Floor Plan")

st.title("Tamil Nadu AI Floor Plan Generator")
st.markdown("Generates NBC, Vastu, Baker, and CMDA/DTCP compliant residential layouts in seconds.")

st.sidebar.header("Input Parameters")
district = st.sidebar.selectbox("District", options=DISTRICTS, index=DISTRICTS.index("Coimbatore"))
plot_w = st.sidebar.number_input("Plot Width (m)", min_value=4.0, max_value=50.0, value=12.0, step=0.5)
plot_d = st.sidebar.number_input("Plot Depth (m)", min_value=8.0, max_value=40.0, value=15.0, step=0.5)
facing = st.sidebar.radio("Plot Facing", options=["North", "South", "East", "West"], index=0)
bhk_val = st.sidebar.selectbox("BHK Configuration", options=["1BHK", "2BHK", "3BHK", "4BHK"], index=1)
floors_str = st.sidebar.radio("Number of Floors", options=["G", "G+1", "G+2"], index=0)
vastu_enabled = st.sidebar.checkbox("Vastu Compliance", value=True)
baker_enabled = st.sidebar.checkbox("Baker Principles", value=True)
seed_input = st.sidebar.number_input("Random Seed (0-999, 0 for Auto)", min_value=0, max_value=999, value=42)

if st.sidebar.button("Generate Floor Plan", type="primary"):
    bhk = int(bhk_val[0])
    floors = 1 if floors_str == "G" else (2 if floors_str == "G+1" else 3)
    facing_chr = facing[0]
    seed_val = int(seed_input) if seed_input > 0 else None
    
    if plot_w * plot_d < 30:
        st.error("Plot area is too small (<30 sqm). Minimum buildable area rules will fail.")
    else:
        with st.spinner("Generating Layout..."):
            try:
                res = generate_plan(
                    plot_w=plot_w, plot_d=plot_d, bhk=bhk, facing=facing_chr, 
                    district=district, floors=floors, vastu=vastu_enabled, 
                    baker=baker_enabled, seed=seed_val
                )
                
                ground_fp = res["ground"]
                render_res = render(ground_fp, output_dir="outputs")
                st.success("Plan generated successfully!")
                
                col_img, col_data = st.columns([2, 1])
                
                with col_img:
                    st.image(render_res["png"], caption="Ground Floor", use_container_width=True)
                    
                    st.markdown("### Downloads")
                    d1, d2 = st.columns(2)
                    with open(render_res["png"], "rb") as f:
                        d1.download_button("Download PNG (A2/150DPI)", f, file_name="floorplan.png", mime="image/png")
                    with open(render_res["dxf"], "rb") as f:
                        d2.download_button("Download DXF (CAD)", f, file_name="floorplan.dxf", mime="application/octet-stream")
                
                with col_data:
                    st.subheader("Compliance Scores")
                    for k, v in res["scores"].items():
                        if "first" not in k:
                            label = k.replace("score_", "").upper()
                            st.progress(float(v), text=f"{label}: {v:.0%}")
                    
                    st.subheader("Room Area Validation")
                    room_data = []
                    for rt, rm in ground_fp.rooms.items():
                        min_a = NBC_MIN_AREA.get(rt, 0.0)
                        room_data.append({"Room": rt.replace("_", " ").title(), "Generated (sqm)": round(rm.area, 2), "NBC Min (sqm)": min_a, "Status": "Pass" if rm.area >= min_a else "Fail"})
                    st.dataframe(pd.DataFrame(room_data), use_container_width=True)
                    
                    st.subheader("Material / Baker Notes")
                    mats = res["metadata"].get("materials", [])
                    if mats:
                        st.json(mats[0] if isinstance(mats[0], dict) else {"Materials": mats})
                    bakers = res["metadata"].get("baker_principles", [])
                    if bakers:
                        st.json(bakers[0] if isinstance(bakers[0], dict) else {"Baker": bakers})

                    st.info(f"**Climate Zone:** {ground_fp.climate_zone}\n\n**Soil Assumption:** {res['metadata']['soil_type']}")
                    st.info(f"**SHAP Top Features Impacting ML Validity:**\n{list(ground_fp.shap_values.keys())[:5] if ground_fp.shap_values else 'None (fast path)'}")
                    
            except Exception as e:
                import traceback
                st.error(f"Error generating plan: {e}")
                st.code(traceback.format_exc())
