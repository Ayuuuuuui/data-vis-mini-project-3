import streamlit as st
import plotly.express as px
import joblib
import pandas as pd
import altair as alt
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import plotly.graph_objects as go

model = joblib.load("model.joblib")

stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

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
  return model.predict([features])[0]


# Sample Thai names and surnames
first_names = ["สมชาย", "วิชัย", "สมศักดิ์", "กิตติ", "อัศวิน", "ประสิทธิ์", "สุริยะ", "ชัยวัฒน์", "วัฒนา", "เอกชัย", "พัฒน์พงศ์", "สุพจน์", "วิเชียร", "อรุณ", "กำธร"]
last_names = ["มีสุข", "สวัสดี", "สุขใจ", "ใจดี", "ใจบุญ", "กิตติกูล", "ชนะพงศ์", "สุวรรณ", "คงเจริญ", "เพิ่มพูน", "เจริญสุข", "ชัยรัตน์", "ทรงชัย", "สุทธิชัย", "รุ่งเรือง"]

# Sample Thai locations
districts = ["สามย่าน", "ลาดพร้าว", "บางนา", "บางเขน", "ห้วยขวาง", "บางกะปิ", "ดอนเมือง", "บางบัวทอง", "บางพลี", "พระโขนง", "พญาไท", "บางกอกน้อย", "บางกอกใหญ่", "ปทุมวัน", "สาทร"]
subdistricts = ["ทุ่งมหาเมฆ", "สวนหลวง", "ลาดยาว", "สีกัน", "บางรัก", "ปากเกร็ด", "บางมด", "ลาดพร้าว", "ศาลายา", "บางกะปิ", "คลองตัน", "พระโขนง", "ลาดพร้าว", "บางนา", "บางซื่อ"]
provinces = ["กรุงเทพฯ", "กทม", "กรุงเทพมหานคร", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ", "นครปฐม", "ชลบุรี", "อยุธยา", "สระบุรี", "ราชบุรี", "อ่างทอง"]
postal_codes = ["10100", "10240", "10120", "10230", "10310", "10150", "10210", "11120", "10270", "10540"]

house_number_variants = [
    lambda: f"{random.randint(1, 999)}/{random.randint(1, 99)}",
    lambda: f"{random.randint(1, 999)}",
    lambda: f"{random.randint(1, 999)}หมู่{random.randint(1, 20)}"]

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


# Function to generate random addresses with shuffled order (except name-surname order)
# Updated function with 'rd' parameter to control shuffling
# def generate_address(num_addresses, seed=42, rd=True):
#     random.seed(seed)
#     addresses = []
#     tags = []
    
#     for _ in range(num_addresses):
#         # Generate name and surname, which must appear together
#         first_name = random.choice(first_names)
#         last_name = random.choice(last_names)
#         name = f"นาย{first_name} {last_name}"
#         name_tag = ["O", "O"]  # Tag 'O' for first name and last name
        
#         # Other components for the address and their tags
#         other_components = [
#             (f"{random.randint(1, 999)}/{random.randint(1, 99)}", "ADDR"),
#             (random.choice(districts), "LOC"),
#             (random.choice(subdistricts), "LOC"),
#             (random.choice(provinces), "LOC"),
#             (random.choice(postal_codes), "POST")
#         ]
        
#         # Conditionally shuffle the other components based on the rd parameter
#         if rd:
#             random.shuffle(other_components)
        
#         # Separate components and their tags (in either shuffled or fixed order)
#         components, component_tags = zip(*other_components)
        
#         # Combine name (first + last) with other components
#         address = f"{name} " + " ".join(components)
#         address_tags = name_tag + list(component_tags)
        
#         # Append results
#         addresses.append(address)
#         tags.append(address_tags)
    
#     return addresses, tags


# def create_df(rd):
#   # Generate 100 random addresses and their tags
#   random_addresses, random_tags = generate_address(100, rd=rd)

#   # Convert the generated addresses and tags to a DataFrame for CSV export
#   df_addresses = pd.DataFrame({
#       "Address": random_addresses,
#       "Tags": random_tags
#   })

#   rows = df_addresses.shape[0]

#   for index, row in df_addresses.iterrows():
#       content = df_addresses.iloc[index,0]
#       df_addresses['Predict'] = df_addresses['Address'].apply(parse)

#   return df_addresses


#---------------------------------------------------
st.set_page_config(
    page_title="NER Visualization",
    page_icon="📊",
    layout="wide"
)

# Create WebApp by Streamlit
st.title('Named Entity Recognition (NER) Visualization')
# Create a function to highlight tags
col1, col2 = st.columns(2)
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

with col2:
    name_format = st.multiselect("Select Name Format", ["นาย", "นาง", "นางสาว","No prefix"], default=["นาย", "นาง", "นางสาว","No prefix"])
    house_number_format = st.multiselect("Select House Number Format", ["123", "123/45", "123หมู่1"], default=["123", "123/45", "123หมู่1"])
    village_format = st.multiselect("Select Village Format", ["หมู่บ้าน", "ม.", "No prefix"], default=["หมู่บ้าน", "ม.", "No prefix"])  # Allow multiple selections
    soi_format = st.multiselect("Select Soi Format", ["ซอย", "ซ.", "No prefix"], default = ["ซอย", "ซ.", "No prefix"])  # Allow multiple selections
    road_format = st.multiselect("Select Road Format", ["ถนน", "ถ.", "No prefix"], default=["ถนน", "ถ.", "No prefix"])  # Allow multiple selections
    subdistrict_format = st.multiselect("Select Subdistrict Format", ["ตำบล", "ต.", "แขวง"], default=["ตำบล", "ต.", "แขวง"])
    district_format = st.multiselect("Select District Format", ["อำเภอ", "อ.", "เขต"], default=["อำเภอ", "อ.", "เขต"])
    province_format = st.multiselect("Select Province Format", ["จังหวัด", "จ.", "No prefix"], default= ["จังหวัด", "จ.", "No prefix"])  # Allow multiple selections

# Modify the variations to use selected format
def generate_address():

    selected_name_format = random.choice(name_format) if name_format else ""
    Name = f"{selected_name_format}{random.choice(first_names)}" if selected_name_format != "No prefix" else random.choice(first_names)

    # Generate house number with randomly selected format
    selected_house_number_format = random.choice(house_number_format) if house_number_format else ""
    if selected_house_number_format == "123":
        house_number = f"{random.randint(1, 999)}"
    elif selected_house_number_format == "123/45":
        house_number = f"{random.randint(1, 999)}/{random.randint(1, 99)}"
    else:
        house_number = f"{random.randint(1, 999)}หมู่{random.randint(1, 20)}"

    # Generate village with randomly selected format
    selected_village_format = random.choice(village_format) if village_format else ""
    village = f"{selected_village_format}{random.choice(village_variants)}" if selected_village_format != "No prefix" else random.choice(village_variants)

    # Generate Soi with randomly selected format
    selected_soi_format = random.choice(soi_format) if soi_format else ""
    soi = f"{selected_soi_format}{random.choice(soi_variants)}" if selected_soi_format != "No prefix" else random.choice(soi_variants)

    # Generate Road with randomly selected format
    selected_road_format = random.choice(road_format) if road_format else ""
    road = f"{selected_road_format}{random.choice(road_variants)}" if selected_road_format != "No prefix" else random.choice(road_variants)

    # Generate Subdistrict with randomly selected format
    selected_subdistrict_format = random.choice(subdistrict_format) if subdistrict_format else ""
    subdistrict = f"{selected_subdistrict_format}{random.choice(subdistrict_variants)}" if selected_subdistrict_format != "No prefix" else random.choice(subdistrict_variants)

    # Generate District with randomly selected format
    selected_district_format = random.choice(district_format) if district_format else ""
    district = f"{selected_district_format}{random.choice(district_variants)}" if selected_district_format != "No prefix" else random.choice(district_variants)

    # Generate Province with randomly selected format
    selected_province_format = random.choice(province_format) if province_format else ""
    province = f"{selected_province_format}{random.choice(province_variants)}" if selected_province_format != "No prefix" else random.choice(province_variants)

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


# Generate samples and predictions

# Define tag labels
tag_labels = {
    "Name": "O",
    "HouseNumber": "ADDR",
    "Village": "ADDR",
    "Soi": "ADDR",
    "Road": "ADDR",
    "Subdistrict": "LOC",
    "District": "LOC",
    "Province": "LOC",
    "PostalCode": "POST"
}

# Update generate_samples function to include labels
def generate_samples():
    sample_addresses = []
    predicted_tags_list = []
    label_list = []

    for _ in range(100):  # Generate 50 samples
        address_data = generate_address()
        customized_address = " ".join([
            address_data[component]
            for component in components_order
            if component_visibility.get(component, False)
        ])
        
        # Collect labels based on visible components
        labels = [
            tag_labels[component]
            for component in components_order
            if component_visibility.get(component, False)
        ]

        sample_addresses.append(customized_address)
        predicted_tags = parse(customized_address)  # NER tags for address
        predicted_tags_list.append(predicted_tags)
        label_list.append(labels)

    return sample_addresses, predicted_tags_list, label_list

with col1:
  # Button to regenerate samples
  if st.button("Generate New Samples"):
      generate_new_samples = True
  else:
      generate_new_samples = False

# Generate or regenerate samples
if (generate_new_samples or 'sample_addresses') and (generate_new_samples or 'label_list') not in st.session_state:
    st.session_state['sample_addresses'], st.session_state['predicted_tags_list'], st.session_state['label_list'] = generate_samples()

sample_addresses = st.session_state['sample_addresses']
predicted_tags_list = st.session_state['predicted_tags_list']
label_list = st.session_state['label_list']


def shuffle_address_components(df):
    random.seed(999)
    shuffled_addresses = []
    shuffled_predictions = []
    shuffled_labels = []

    # Iterate over each row in the original DataFrame
    for i in range(len(df)):
        # Split address into tokens
        address_tokens = df["Address"].iloc[i].split()
        prediction_tags = df["Prediction"].iloc[i]
        label_tags = df["Labels"].iloc[i]

        # Pair tokens with their corresponding prediction and label tags
        token_data = list(zip(address_tokens, prediction_tags, label_tags))

        # Shuffle the token, prediction, and label pairs
        random.shuffle(token_data)

        # Separate tokens, predictions, and labels back into separate lists
        shuffled_tokens, shuffled_pred, shuffled_lbl = zip(*token_data)

        # Join tokens to form the shuffled address string
        shuffled_address = " ".join(shuffled_tokens)

        # Append the shuffled data to their respective lists
        shuffled_addresses.append(shuffled_address)
        shuffled_predictions.append(parse(shuffled_address))
        shuffled_labels.append(list(shuffled_lbl))

    # Create a new DataFrame with the shuffled addresses and tags
    shuffled_df = pd.DataFrame({
        "Address": shuffled_addresses,
        "Prediction": shuffled_predictions,
        "Labels": shuffled_labels
    })

    return shuffled_df

# Display DataFrame with addresses, predicted tags, and labels
df_addresses = pd.DataFrame({
    "Address": sample_addresses,
    "Prediction": predicted_tags_list,
    "Labels": label_list
})

df_shuffled_addresses = shuffle_address_components(df_addresses)

st.write("### Table Address Prediction")
st.dataframe(df_addresses, use_container_width=False)


def create_confusion_matrix(df_addresses):
  # Flatten the 'Tags' and 'Predict' columns to compare corresponding elements
  true_tags = [tag for tags in df_addresses["Labels"] for tag in tags]
  predicted_tags = [tag for tags in df_addresses["Prediction"] for tag in tags]

  # Create the confusion matrix
  cm = confusion_matrix(true_tags, predicted_tags)

  # Get the unique labels (tags) to display the confusion matrix with proper labels
  labels = np.unique(true_tags)

  # Convert the confusion matrix into a DataFrame for better visualization
  cm_df = pd.DataFrame(cm, index=labels, columns=labels)

  return cm_df

st.subheader('Model Performance')

def highlight_address(address, tags):
    highlighted_address = ""
    tag_colors = {
        "O": "background-color: #FFB067; border-radius: 5px; padding: 2px;",
        "LOC": "background-color: #FFED86; border-radius: 5px; padding: 2px;",
        "POST": "background-color: #A2DCE7; border-radius: 5px; padding: 2px;",
        "ADDR": "background-color: #F8CCDC; border-radius: 5px; padding: 2px;"
    }
    
    words = address.split()
    for word, tag in zip(words, tags):
        style = tag_colors.get(tag, "")
        highlighted_address += f"<span style='{style}'>{word}</span> "
    
    return highlighted_address

# def get_random_ex(df_addresses):
#     return df_addresses.sample(n=1).iloc[0]



tab1, tab2 = st.tabs(['Confusion Matrix','Bar Chart'])
with tab1:
  col3, col4 = st.columns(2)

  with col3:
    st.markdown('##### Confusion Matrix (Shuffled Position)')
    st.caption('Example Prediction')

    sample_address = df_shuffled_addresses.iloc[22,:] # just an example
    address = sample_address[0]
    tags = sample_address[1]

    # Highlight the example address
    highlighted_example = highlight_address(address, tags)
    # Streamlit markdown with the example and legend
    st.markdown(
        f"""
        {highlighted_example}
        """,
        unsafe_allow_html=True
      )


    # Legend to explain each tag
    st.markdown(
        """
        ###### Legend:
        <span style='background-color: #FFB067; border-radius: 5px; padding: 2px;'>O</span>
        <span style='background-color: #FFED86; border-radius: 5px; padding: 2px;'>LOC</span>
        <span style='background-color: #A2DCE7; border-radius: 5px; padding: 2px;'>POST</span>
        <span style='background-color: #F8CCDC; border-radius: 5px; padding: 2px;'>ADDR</span>
        """,
        unsafe_allow_html=True
    )

    # Plotting the confusion matrix using Seaborn and Matplotlib
    with st.container(border = True):
      # Display the plot within a specific div container
      cm_df_rand = create_confusion_matrix(df_shuffled_addresses)
      fig, ax = plt.subplots(figsize=(8, 6))  # You can still control fig size
      sns.heatmap(cm_df_rand, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
      
      # Set plot labels and title
      ax.set_xlabel('Predicted Labels')
      ax.set_ylabel('True Labels')
      # ax.set_title('Confusion Matrix')
      # Display the plot in Streamlit with the custom style class
      st.pyplot(fig)

    st.dataframe(df_shuffled_addresses, use_container_width=False)


  with col4:
    st.markdown('##### Confusion Matrix (Fixed Position)')
    st.caption('Example Prediction')

    sample_address = df_addresses.iloc[22,:] # just an example
    address = sample_address[0]
    tags = sample_address[1]

    # Highlight the example address
    highlighted_example = highlight_address(address, tags)
    # Streamlit markdown with the example and legend
    st.markdown(
        f"""
        {highlighted_example}
        """,
        unsafe_allow_html=True
      )
    
    # Legend to explain each tag
    st.markdown(
        """
        ###### Legend:
        <span style='background-color: #FFB067; border-radius: 5px; padding: 2px;'>O</span>
        <span style='background-color: #FFED86; border-radius: 5px; padding: 2px;'>LOC</span>
        <span style='background-color: #A2DCE7; border-radius: 5px; padding: 2px;'>POST</span>
        <span style='background-color: #F8CCDC; border-radius: 5px; padding: 2px;'>ADDR</span>
        """,
        unsafe_allow_html=True
    )

    # Plotting the confusion matrix using Seaborn and Matplotlib
    with st.container(border = True):
      # Display the plot within a specific div container
      cm_df_fixed = create_confusion_matrix(df_addresses)
      fig, ax = plt.subplots(figsize=(8, 6))  # You can still control fig size
      sns.heatmap(cm_df_fixed, annot=True, fmt="d", cmap="Reds", cbar=True, ax=ax)
      
      # Set plot labels and title
      ax.set_xlabel('Predicted Labels')
      ax.set_ylabel('True Labels')
      # ax.set_title('Confusion Matrix')
      # Display the plot in Streamlit with the custom style class
      st.pyplot(fig)

    st.dataframe(df_addresses, use_container_width=False)

  with tab2:
    st.write('test')



tag_colors = {
    "O": "#FFB067",
    "LOC": "#FFED86",
    "POST": "#F8CCDC",
    "ADDR": "#A2DCE7"
}

#Sankey Diagram
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
        pad=20,
        thickness=20,
        line=dict(color="rgba(0,0,0,0)", width=0),
        label=labels,
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color= '#EEEDE7'
    )
))

# Adjust layout for minimalistic font and larger size
fig.update_layout(
      font=dict(
          family = "Arial",  # Minimalist font style
          size = 16,  # Larger font size
      ),
      width=2000,  # Adjust width as needed
      height=500   # Adjust height as needed
  )

# Display Sankey Diagram in Streamlit
st.plotly_chart(fig, use_container_width=False)
