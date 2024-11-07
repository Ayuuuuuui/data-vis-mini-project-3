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

# Function to generate random addresses with shuffled order (except name-surname order)
def generate_address(num_addresses, seed=42):
    random.seed(seed)
    addresses = []
    tags = []
    
    for _ in range(num_addresses):
        # Generate name and surname, which must appear together
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        name = f"นาย{first_name} {last_name}"
        name_tag = ["O", "O"]  # Tag '0' for first name and 'O' for last name
        
        # Other components for the address and their tags
        other_components = [
            (f"{random.randint(1, 999)}/{random.randint(1, 99)}", "ADDR"),
            (random.choice(districts), "LOC"),
            (random.choice(subdistricts), "LOC"),
            (random.choice(provinces), "LOC"),
            (random.choice(postal_codes), "POST")
        ]
        
        # Shuffle the other components
        random.shuffle(other_components)
        
        # Separate components and their tags after shuffling
        shuffled_components, shuffled_tags = zip(*other_components)
        
        # Combine name (first + last) with other shuffled components
        address = f"{name} " + " ".join(shuffled_components)
        address_tags = name_tag + list(shuffled_tags)
        
        # Append results
        addresses.append(address)
        tags.append(address_tags)
    
    return addresses, tags

# Generate 100 random addresses and their tags
random_addresses, random_tags = generate_address(100)

# Convert the generated addresses and tags to a DataFrame for CSV export
df_addresses = pd.DataFrame({
    "Address": random_addresses,
    "Tags": random_tags
})

rows = df_addresses.shape[0]

for index, row in df_addresses.iterrows():
    content = df_addresses.iloc[index,0]
    df_addresses['Predict'] = df_addresses['Address'].apply(parse)


# Flatten the 'Tags' and 'Predict' columns to compare corresponding elements
true_tags = [tag for tags in df_addresses["Tags"] for tag in tags]
predicted_tags = [tag for tags in df_addresses["Predict"] for tag in tags]

# Create the confusion matrix
cm = confusion_matrix(true_tags, predicted_tags)

# Get the unique labels (tags) to display the confusion matrix with proper labels
labels = np.unique(true_tags)

# Convert the confusion matrix into a DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=labels, columns=labels)


#---------------------------------------------------
st.set_page_config(
    page_title="NER Visualization",
    page_icon="📊",
    layout="wide"
)

# Create WebApp by Streamlit
st.title('Named Entity Recognition (NER) Visualization')
# Create a function to highlight tags

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


# def get_random_ex():
#     # Select a random number from the fixed set without setting a seed for the choice
#     return random.choice(df_addresses.values)

def get_random_ex():
    return df_addresses.sample(n=1).iloc[0]

st.subheader('Random position tags')
# Random and Displaythe highlighted address in Streamlit
if st.button('Generate Example'):
  # Example of an address with tags to highlight
  sample_address = get_random_ex()

  address = sample_address[0]
  tags = sample_address[1]

  # Highlight the example address
  highlighted_example = highlight_address(address, tags)
  # Streamlit markdown with the example and legend
  st.markdown(
      f"""
      ### Example Address with Highlighted Tags
      {highlighted_example}
      """,
      unsafe_allow_html=True
    )


  # Legend to explain each tag
  st.markdown(
      """
      ### Legend:
      <span style='background-color: #FFB067; border-radius: 5px; padding: 2px;'>O</span>
      <span style='background-color: #FFED86; border-radius: 5px; padding: 2px;'>LOC</span>
      <span style='background-color: #A2DCE7; border-radius: 5px; padding: 2px;'>POST</span>
      <span style='background-color: #F8CCDC; border-radius: 5px; padding: 2px;'>ADDR</span>
      """,
      unsafe_allow_html=True
  )

col1, col2 = st.columns(2)

with col1:
  st.subheader('Confusion Matrix from Testing data (Random Pos)')

  # Plotting the confusion matrix using Seaborn and Matplotlib
  with st.container(border = True):
    # Display the plot within a specific div container
    fig, ax = plt.subplots(figsize=(8, 6))  # You can still control fig size
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    
    # Set plot labels and title
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    # ax.set_title('Confusion Matrix')
    # Display the plot in Streamlit with the custom style class
    st.pyplot(fig)

  st.dataframe(df_addresses, use_container_width=False)


with col2:
  st.subheader('Confusion Matrix from Testing data (Fixed Pos)')

  # Plotting the confusion matrix using Seaborn and Matplotlib
  with st.container(border = True):
    # Display the plot within a specific div container
    fig, ax = plt.subplots(figsize=(8, 6))  # You can still control fig size
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Reds", cbar=True, ax=ax)
    
    # Set plot labels and title
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    # ax.set_title('Confusion Matrix')
    # Display the plot in Streamlit with the custom style class
    st.pyplot(fig)

  st.dataframe(df_addresses, use_container_width=False)