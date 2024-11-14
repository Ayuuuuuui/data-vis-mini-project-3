# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import joblib
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

st.set_page_config(
    page_title="NER Visualization",
    page_icon="📊",
    layout="wide"
)
# Load model
model = joblib.load("model.joblib")

# Define Thai stopwords
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

# Streamlit UI
st.title("Customizable Address Generator with NER")

# Sample Thai names and locations
first_names = ["สมชาย", "วิชัย", "สมศักดิ์"]
last_names = ["มีสุข", "สุขใจ", "ใจดี"]
districts = ["สามย่าน", "ลาดพร้าว", "บางนา"]
subdistricts = ["ทุ่งมหาเมฆ", "สวนหลวง", "ลาดยาว"]
provinces = ["กรุงเทพฯ", "นนทบุรี", "ปทุมธานี"]
postal_codes = ["10110", "10120", "10150", "10160", "10170", "10240", "10250", "10260", "10310", "10400",
                "11000", "11120", "11130","12000", "12120", "12130","10540", "10560", "10570"]

house_number_variants = [
    lambda: f"{random.randint(1, 999)}/{random.randint(1, 99)}",
    lambda: f"{random.randint(1, 999)}",
    lambda: f"{random.randint(1, 999)} หมู่ {random.randint(1, 20)}"]

village_variants = [ "รักนิยม","ปิยะ","เพชรเกษม","ท่าช้าง","สวนลุม",
                    "นครทอง","อรุณสวัสดิ์","อัมพร","คลองสี่", "บางแค"]

soi_variants = ["สาทร11", "รามคำแหง24","สุขุมวิท39","อ่อนนุช18","พัฒนาการ20","นวลจันทร์","ทองหล่อ23"]

road_variants = ["สาทร", "สุขุมวิท", "รามคำแหง", "พัฒนาการ", "วิภาวดีรังสิต","อ่อนนุช", 
                 "ทองหล่อ", "พระราม9", "ลาดพร้าว"]

subdistrict_variants = ["ทุ่งมหาเมฆ", "สวนหลวง", "ลาดยาว", "สีกัน", "บางรัก", "ปากเกร็ด", 
                        "บางมด", "ลาดพร้าว", "ศาลายา", "บางกะปิ", "คลองตัน", "พระโขนง", 
                        "ลาดพร้าว", "บางนา", "บางซื่อ"]

district_variants = ["สามย่าน", "ลาดพร้าว", "บางนา", "บางเขน", "ห้วยขวาง", "บางกะปิ", "ดอนเมือง", 
                     "บางบัวทอง", "บางพลี", "พระโขนง", "พญาไท", "บางกอกน้อย", "บางกอกใหญ่", "ปทุมวัน", "สาทร"]

province_variants = ["ราชบุรี","กรุงเทพ","กรุงเทพมหานคร"]

# Function for NER tagging prediction
def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True
    return features

def parse(text):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    prediction = model.predict([features])[0]
    return prediction

col1, col3 = st.columns(2)
with col3:
    name_format = st.multiselect("Select Name Format", ["นาย", "นาง", "นางสาว","ไม่มี"], default=["นาย", "นาง", "นางสาว","ไม่มี"])
    house_number_format = st.multiselect("Select House Number Format", ["123", "123/45", "123 หมู่ 1"], default=["123", "123/45", "123 หมู่ 1"])
    village_format = st.multiselect("Select Village Format", ["หมู่บ้าน", "ม.", "ไม่มี"], default=["หมู่บ้าน", "ม.", "ไม่มี"])  # Allow multiple selections
    soi_format = st.multiselect("Select Soi Format", ["ซอย", "ซ.", "ไม่มี"], default = ["ซอย", "ซ.", "ไม่มี"])  # Allow multiple selections
    road_format = st.multiselect("Select Road Format", ["ถนน", "ถ.", "ไม่มี"], default=["ถนน", "ถ.", "ไม่มี"])  # Allow multiple selections
    subdistrict_format = st.multiselect("Select Subdistrict Format", ["ตำบล", "ต.", "แขวง"], default=["ตำบล", "ต.", "แขวง"])
    district_format = st.multiselect("Select District Format", ["อำเภอ", "อ.", "เขต"], default=["อำเภอ", "อ.", "เขต"])
    province_format = st.multiselect("Select Province Format", ["จังหวัด", "จ.", "ไม่มี"], default= ["จังหวัด", "จ.", "ไม่มี"])  # Allow multiple selections

# Modify the variations to use selected format
def generate_address():

    selected_name_format = random.choice(name_format) if name_format else ""
    Name = f"{selected_name_format}{random.choice(first_names)}" if selected_name_format != "ไม่มี" else random.choice(first_names)

    # Generate house number with randomly selected format
    selected_house_number_format = random.choice(house_number_format) if house_number_format else ""
    if selected_house_number_format == "123":
        house_number = f"{random.randint(1, 999)}"
    elif selected_house_number_format == "123/45":
        house_number = f"{random.randint(1, 999)}/{random.randint(1, 99)}"
    else:
        house_number = f"{random.randint(1, 999)} หมู่ {random.randint(1, 20)}"

    # Generate village with randomly selected format
    selected_village_format = random.choice(village_format) if village_format else ""
    village = f"{selected_village_format}{random.choice(village_variants)}" if selected_village_format != "ไม่มี" else random.choice(village_variants)

    # Generate Soi with randomly selected format
    selected_soi_format = random.choice(soi_format) if soi_format else ""
    soi = f"{selected_soi_format}{random.choice(soi_variants)}" if selected_soi_format != "ไม่มี" else random.choice(soi_variants)

    # Generate Road with randomly selected format
    selected_road_format = random.choice(road_format) if road_format else ""
    road = f"{selected_road_format}{random.choice(road_variants)}" if selected_road_format != "ไม่มี" else random.choice(road_variants)

    # Generate Subdistrict with randomly selected format
    selected_subdistrict_format = random.choice(subdistrict_format) if subdistrict_format else ""
    subdistrict = f"{selected_subdistrict_format}{random.choice(subdistrict_variants)}" if selected_subdistrict_format != "ไม่มี" else random.choice(subdistrict_variants)

    # Generate District with randomly selected format
    selected_district_format = random.choice(district_format) if district_format else ""
    district = f"{selected_district_format}{random.choice(district_variants)}" if selected_district_format != "ไม่มี" else random.choice(district_variants)

    # Generate Province with randomly selected format
    selected_province_format = random.choice(province_format) if province_format else ""
    province = f"{selected_province_format}{random.choice(province_variants)}" if selected_province_format != "ไม่มี" else random.choice(province_variants)

    # Combine all components into the final address
    address_components = {
        "Name" : Name,
        "HouseNumber": house_number,
        "Village": village,
        "Soi": soi,
        "Road": road,
        "Subdistrict": subdistrict,
        "District": district,
        "Province": province,
        "PostalCode": random.choice(postal_codes)
    }
    return address_components


# Generate and display the address
#if st.button("Generate Address"):
#    address_data = generate_address()
#    # Join address components that are not empty
#    address = " ".join([value for value in address_data.values() if value])
#    st.write("Generated Address:", address)

#col1, col2 = st.columns(2)

with col1:
# Address component selection
    components_order = st.multiselect(
        "Select components and their order for the address:",
        options=["Name", "HouseNumber", "Village", "Soi", "Road", "Subdistrict", "District", "Province", "PostalCode"],
        default=["Name", "HouseNumber", "Village", "Soi", "Road", "Subdistrict", "District", "Province", "PostalCode"]
    )

    # Address component visibility
    component_visibility = {
        "Name": st.checkbox("Include Name", True),
        "HouseNumber": st.checkbox("Include House Number", True),
        "Village": st.checkbox("Include Village", True),
        "Soi": st.checkbox("Include Soi", True),
        "Road": st.checkbox("Include Road", True),
        "Subdistrict": st.checkbox("Include Subdistrict", True),
        "District": st.checkbox("Include District", True),
        "Province": st.checkbox("Include Province", True),
        "PostalCode": st.checkbox("Include Postal Code", True)
    }

# Generate samples and predictions
def generate_samples():
    sample_addresses = []
    predicted_tags_list = []
    for _ in range(50):  # You can adjust the number of samples here
        address_data = generate_address()
        customized_address = " ".join([
            address_data[component]
            for component in components_order
            if component_visibility.get(component, False)
        ])
        sample_addresses.append(customized_address)
        predicted_tags = parse(customized_address)
        predicted_tags_list.append(predicted_tags)
    return sample_addresses, predicted_tags_list
# Button to regenerate samples
if st.button("Generate New Samples"):
    generate_new_samples = True
else:
    generate_new_samples = False

# Generate or regenerate samples
if generate_new_samples or 'sample_addresses' not in st.session_state:
    st.session_state['sample_addresses'], st.session_state['predicted_tags_list'] = generate_samples()

sample_addresses = st.session_state['sample_addresses']
predicted_tags_list = st.session_state['predicted_tags_list']

# Display DataFrame with addresses and predicted tags
df_addresses = pd.DataFrame({
    "Address": sample_addresses,
    "Predicttion": predicted_tags_list
})
st.write("### Table Address Prediction")
st.dataframe(df_addresses)

# Define custom colors for each tag
tag_colors = {
    "O": "#ffff00",
    "LOC": "#ff00ff",
    "POST": "#00ffff",
    "ADDR": "#00ff00"
}

# Sankey Diagram
st.write("### Sankey Diagram of Prediction Flows")

# Split tags into individual levels
df_tags_split = pd.DataFrame(predicted_tags_list, columns=[f"Level {i+1}" for i in range(max(len(tags) for tags in predicted_tags_list))])

# Prepare data for Sankey Diagram
levels = df_tags_split.columns
unique_tags = ["O", "LOC", "POST", "ADDR"]
labels = [f"{tag} - {level}" for level in levels for tag in unique_tags]
label_map = {label: i for i, label in enumerate(labels)}

# Assign colors to nodes based on tag_colors
node_colors = [tag_colors[tag.split(" - ")[0]] for tag in labels]

source = []
target = []
value = []

# Create flows between consecutive levels based on tag transitions
for i in range(len(levels) - 1):
    level1 = df_tags_split[levels[i]]
    level2 = df_tags_split[levels[i + 1]]
    flow_data = pd.concat([level1, level2], axis=1).value_counts().reset_index()
    for (src_tag, tgt_tag), count in zip(flow_data.values[:, :2], flow_data.values[:, 2]):
        src_label = f"{src_tag} - {levels[i]}"
        tgt_label = f"{tgt_tag} - {levels[i + 1]}"
        if src_label in label_map and tgt_label in label_map:
            source.append(label_map[src_label])
            target.append(label_map[tgt_label])
            value.append(count)

# Create Sankey Diagram with custom colors and font adjustments
fig = go.Figure(go.Sankey(
    node=dict(
        pad=10,
        thickness=20,
        #line=dict(color="#F0F2F6", width=0.5),
        label=labels,
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
))

# Adjust layout for minimalistic font and larger size
fig.update_layout(
    font=dict(
        family="Arial, sans-serif",  # Minimalist font style
        size=12,  # Larger font size
        color="black"   
    ),
    width=2000,  # Adjust width as needed
    height=500   # Adjust height as needed
)

# Display Sankey Diagram in Streamlit
st.plotly_chart(fig, use_container_width=False)
