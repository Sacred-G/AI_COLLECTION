import os
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from st_multimodal_chatinput import multimodal_chatinput

# Set Google API key
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

# Make model
model = genai.GenerativeModel('gemini-pro')
vision_model = ChatGoogleGenerativeAI(model="gemini-pro-vision")

# Set page configuration
st.set_page_config(
    page_title="Gemini Vision Image Information",
    page_icon=":bridge_at_night:",
    layout="wide",
)
st.image("img/Robot.jpg",width=300)

# Add header image to the sidebar
header_image_url = "https://miro.medium.com/v2/resize:fit:720/format:webp/0*mzH1auQ-6FbZ9pUB.jpg"
st.sidebar.image(header_image_url, width=200)

# Introduction in the main content
st.title("Gemini Vision Image Analysis")

# Main app content
st.sidebar.markdown("""
   👀 Welcome to the Gemini Vision Image Information app! 🌌✨

Explore the power of Gemini Vision as you upload an image and unlock its secrets. Whether it's a stunning photograph, a piece of art, or a snapshot from your collection, Gemini Vision will analyze the image and provide you with detailed information.

How it works:
1. Upload an image using the file uploader.
2. Wait for Gemini Vision to work its magic.
3. Receive a detailed description and insights about the content of your image.
4. Discover the hidden gems within your visuals and delve into the fascinating details revealed by Gemini Vision. Unleash the potential of image analysis with just a few clicks!

**Example:**

 Imagine uploading a snapshot of your grandma's secret lasagna or that gourmet pizza you enjoyed on vacation. Gemini Vision goes beyond visual recognition to provide you with insights and culinary inspiration.

  ❤️ Upload a picture of your homemade pizza! ❤️
⭐️Gemini Vision reveals the types of cheese, toppings, and even suggests a pizza dough ⭐️
"""
)

# Having trouble with chat input box, always in the middle of chat https://github.com/het-25/st-multimodal-chatinput
def reconfig_chatinput():
    st.sidebar.markdown(
        """
  <style>
  {
          width: 100%; /* Span the full width of the viewport */;
          background-color: #FFC0CB;
          }
  </style>
  """,
        unsafe_allow_html=True,
    )

reconfig_chatinput()

# Seed msg, init chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask Gemini anything || send pic w/ a prompt!"
        }
    ]

# Display chat msgs from history upon rerun
for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]):
        st.sidebar.markdown(msg["content"])

# Process, store query + resp
def call_llm_just_text(q):
    response = model.generate_content(q)

    # Display assistant msg
    with st.sidebar.chat_message("assistant", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGVfLdUg7kVxuSqqBGgAL3UJeQgRCLPhxIZlXbVxmUAdYaJm-fcUal7x-FHhwxzpeg6_M&usqp=CAU"):
        st.sidebar.markdown(response.text)

    # Store user msg
    st.session_state.messages.append(
        {
            "role": "user",
            "content": q
        }
    )

    # Store assistant msg
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.text
        }
    )

# Process, store query + resp
def call_llm_with_img(q):
    with st.sidebar:
        st.spinner('Processing📈...')   
        uploaded_images = q["images"]  # List of base64 encodings of uploaded imgs
        txt_inp = q["text"]  # Submitted text
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": txt_inp,
                },
                {
                    "type": "image_url",
                    "image_url": uploaded_images[0]
                },
            ]
        )
        resp = vision_model.invoke([msg])

    # Display assistant msg
    with st.sidebar.chat_message("assistant", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGVfLdUg7kVxuSqqBGgAL3UJeQgRCLPhxIZlXbVxmUAdYaJm-fcUal7x-FHhwxzpeg6_M&usqp=CAU"):
        st.sidebar.markdown(resp.content)

    # Store user msg
    st.session_state.messages.append(
        {
            "role": "user",
            "content": txt_inp
        }
    )

    # Store assistant msg
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resp.content
        }
    )

user_inp = multimodal_chatinput()  # Multimodal streamlit chat input in the sidebar

# Call llm funcs when input is given
if user_inp:
    # Display user msg
    with st.sidebar.chat_message("user", avatar="https://d112y698adiu2z.cloudfront.net/photos/production/user_photos/000/517/300/datas/profile.png"):
        st.sidebar.markdown(user_inp["text"])

    # Check if just text
    if len(user_inp["images"]) == 0:
        call_llm_just_text(user_inp["text"])

    # User_inp includes image
    else:
        call_llm_with_img(user_inp)

st.sidebar.markdown(
    """
    <div>
    <p style="text-align: center;font-family:Arial; color:Pink; font-size: 12px;display: table-cell; vertical-align: bottom">I ❤️ Tonya Kilgore for putting up with me while making this app.✅ out <a href="https://www.stevenbouldin.com">My personal website</a></p>
    </div>
    """,
    unsafe_allow_html=True,  # HTML tags found in body escaped -> treated as pure text
)