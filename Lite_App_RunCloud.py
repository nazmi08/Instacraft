# Third-party library imports
from dotenv import load_dotenv
from PIL import Image
import openai

# import torch

# Diffusers library imports
# from diffusers import (
#     AutoPipelineForText2Image,
#     StableDiffusionPipeline,
#     EulerDiscreteScheduler,
#     StableDiffusionImg2ImgPipeline,
#     AutoPipelineForImage2Image,
# )

# Google AI API
import google.generativeai as genai
import openai

# Streamlit framework import
import streamlit as st

# Load .env file
load_dotenv()

# Initialize API client
openai.api_key = st.secrets['OPENAI_API_KEY']
genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])

artist_images = {
    "None": "",
    "Claude Monet": [
        "https://dreamsinparis.com/wp-content/uploads/2022/05/Water-Lilies-by-Claude-Monet-2-1024x768.jpg",
        "https://dreamsinparis.com/wp-content/uploads/2022/05/Impression-Sunrise-by-Claude-Monet2-1024x786.jpg",
        "https://dreamsinparis.com/wp-content/uploads/2022/07/Haystacks-by-Monet-1024x786.jpg",
        "https://dreamsinparis.com/wp-content/uploads/2022/07/Wild-Poppies-near-Argenteuil-by-Claude-Monet-1024x780.jpg",
        "https://dreamsinparis.com/wp-content/uploads/2022/07/Poplars-by-Claude-Monet-815x1024.jpg"
    ],
    "Vincent van Gogh": [
        "https://www.artst.org/wp-content/uploads/2020/12/Van-Gogh-Starry-Night.jpg.webp",
        "https://www.artst.org/wp-content/uploads/2021/09/The-Potato-Eaters.jpg.webp",
        "https://art-facts.com/wp-content/uploads/2022/04/Anxiety-Edvard-Munch.jpg.webp",
        "https://www.artst.org/wp-content/uploads/2020/06/Cafe-Terrace-at-Night.jpg.webp",
        "https://www.artst.org/wp-content/uploads/2020/06/sunflowers.jpg.webp"
    ],
    "Edvard Munch": [
        "https://art-facts.com/wp-content/uploads/2022/04/Famous-Edvard-Munch-paintings-The-Scream-768x866.jpg.webp",
        "https://art-facts.com/wp-content/uploads/2022/04/The-Sick-Child-Edvard-Munch.jpg.webp",
        "https://art-facts.com/wp-content/uploads/2022/04/Anxiety-Edvard-Munch.jpg.webp",
        "https://art-facts.com/wp-content/uploads/2022/04/The-Dance-of-Life-Edvard-Munch-768x502.jpg.webp",
        "https://art-facts.com/wp-content/uploads/2022/04/The-Yellow-Log-Edvard-Munch-768x623.jpg.webp"
    ],
    "Georgia O'Keeffe": [
        "https://artisticjunkie.com/wp-content/uploads/2018/10/Georgia-O-Keeffe-Flower-Paintings.jpg",
        "https://artisticjunkie.com/wp-content/uploads/2018/10/Georgia-O-Keeffe-Flowers.jpg",
        "https://artisticjunkie.com/wp-content/uploads/2018/10/Georgia-O-Keeffe-Paintings.jpg",
        "https://artisticjunkie.com/wp-content/uploads/2018/10/Georgia-O-Keeffe-Red-Poppy-768x602.jpg",
        "https://artisticjunkie.com/wp-content/uploads/2018/10/Cow-Skull-Georgia-O-Keeffe-Best-Paintings.jpg"
    ],
    "Gustave Klimt": [
        "https://cdn.thecollector.com/wp-content/uploads/2023/01/gustav-klimt-the-kiss-painting-1.jpg?width=828&quality=55",
        "https://cdn.thecollector.com/wp-content/uploads/2023/01/gustav-klimt-philosophy-university-of-vienna-painting.jpg?width=600&quality=55",
        "https://cdn.thecollector.com/wp-content/uploads/2023/01/gustav-klimt-life-and-death-painting-1.jpg?width=1080&quality=55",
        "https://cdn.thecollector.com/wp-content/uploads/2023/01/gustav-klimt-portrait-of-emilie-floge.jpg?width=384&quality=55",
        "https://cdn.thecollector.com/wp-content/uploads/2023/01/gustav-klimt-expectation-stoclet-frieze.jpg?width=480&quality=55"
    ]
}

input_images = {
    "None": "",
    "Claude Monet":"https://i.imgur.com/MvXpcc6.jpeg",
    "Vincent van Gogh":"https://i.imgur.com/pPYxIwO.jpg",
    "Edvard Munch": "https://i.imgur.com/YjICv7V.jpg",
    "Georgia O'Keeffe": "https://i.imgur.com/g8xKAjF.jpg",
    "Gustave Klimt":"https://i.imgur.com/qx0QQzR.jpg "
}
output_images = {
    "None": "",
    "Claude Monet":"https://i.imgur.com/ICyWKXe.jpg",
    "Vincent van Gogh":"https://i.imgur.com/lDBTXhv.jpg",
    "Edvard Munch": "https://i.imgur.com/Iwrm7w6.jpg",
    "Georgia O'Keeffe": "https://i.imgur.com/n4wi1uS.jpg",
    "Gustave Klimt": "https://i.imgur.com/tCdOQxo.jpg"
}

page_bg_img = """
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.pexels.com/photos/1029618/pexels-photo-1029618.jpeg");
        background-size: cover;
    }
    [data-testid="stSidebarContent"] {
        background-color: #c6d6d4
    }
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }
    [data-testid="stExpanderDetails"]{
        background-color: #f3f0eb
    }
    [data-testid="stExpander"]{
        background-color: #bed7dc
    }
</style>
"""
st.set_page_config(
    page_title = 'Reimagine AI',
    page_icon = '🖼️'
)
st.markdown(page_bg_img, unsafe_allow_html=True)

def generate_dalle(msg, image_size=256, artist="None", num_images=1):
    if artist == "None":
        prompt=f'{msg} in random style'
    else:
        prompt=f'{msg} in {artist} artstyle'
    response = openai.images.generate(
        model='dall-e-2',
        prompt=prompt,
        n=num_images,
        size=f'{image_size}x{image_size}',
        quality='standard'
    )
    image_url = response.data[0].url
    return image_url

def generate_content(uploaded_file):
    image = Image.open(uploaded_file)
    model_1 = "gemini-pro-vision"
    vision_model = genai.GenerativeModel(model_1)
    prompt = "What image is this, including:** \n" \
            "  * The main object(s) in the foreground and their characteristics.\n" \
            "  * The background setting and any prominent elements.\n" \
            "  * Any actions or interactions taking place in the scene.\n" \
            "  * The overall mood or atmosphere of the image.\n"\
            " Total word in response must strictly be less than 40 words."
    response = vision_model.generate_content([prompt,image])
    return response.text

def display_artist_images(artist_name, image_urls):
  if artist_name != None:
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, image_url in enumerate(image_urls):
        try:
            col = [col1, col2, col3, col4, col5][i % 5]
            col.image(image_url, use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

def display_sample_images(artist_name, input_images, output_images):
  input_image_url = input_images.get(artist_name, "")
  output_image_url = output_images.get(artist_name, "")
  if input_image_url and output_image_url:
    col1, col2 = st.columns(2)
    col1.write("Input:")
    col1.image(input_image_url, use_column_width=True)
    col2.write("Output:")
    col2.image(output_image_url, use_column_width=True)
  else:
    st.write("No information available for this artist.")

choice = st.sidebar.selectbox("Go to:", ["Home", "All Generated Images"])

if choice == "Home":
    st.title("InstaCraft - Unleash Your Creative Vision")
    with st.expander("**About the app**", expanded=False):
        st.write(    
    """
        **InstaCraft** is AI Art Generator with powerful tool to **transform your ideas into stunning visuals.** 
        It utilizes cutting-edge AI to **instantly apply diverse artistic styles** to your images, empowering you to 
        **explore creative possibilities** and **visualize your concepts** in an engaging and impactful way.\n
        **Benefits:**\n
        - **Spark creative inspiration:** Generate unique image variations based on your chosen artistic style.
        - **Enhance individual expression:** Personalize your images with an artistic flair.
        - **Elevate marketing materials:** Capture attention and create visually compelling marketing materials.
        - **Showcase tourist attractions:** Generate visually striking representations of tourist destinations.\n
        **IMPORTANT NOTICE:** 
          - The generated image showcase in example is generated using SDv2 & SDXL, due to nature using SD must 
        having GPU or using API services, it wouldn't be able to run when deploy it into streamlit cloud.
        Therefore, only DALL-E & Gemini were used in the deployment streamlit cloud.\n
          - Gemini's description can make mistakes. Therefore, the image generated will not be directly similar to image uploaded.\n
          - Check out my Github [Instacraft](https://github.com/nazmi08/Instacraft) 
            for further details on the applications and to get the advanced applications that includes SDv2 and SDXL so you can run it locally on the computer.
    """)
    with st.expander("**How to use the app**", expanded=True):
        st.write(    
    """
        **1. Upload an Image:** Click "Browse files" or drag and drop your image into the designated area.\n
        **2. Click "Describe Image" and wait for the process to finish.** \n
        *The button will show under the image and the generated description will show up in the main content page.\n
        **3. Optional:** Generate images based on the description.
        - Click "Dall-E 2".
        - Choose your desired image size and number of images.
        - Click "Generate Image" to receive the results.
    """)
    artist_name = st.selectbox("Pick your artist for the image generated artstyle:", tuple(artist_images.keys()), index=0)
    st.session_state["artist_name"] = artist_name
    if artist_name in artist_images:
        if artist_name != "None":
            st.write(f"{artist_name}'s Art")
            display_artist_images(artist_name, artist_images[artist_name])
            with st.expander("**Example generated image based on the artist's artstyle:**", expanded=True):
                display_sample_images(artist_name, input_images, output_images)
    else:
        st.info("Artist not found.")        

if choice == 'Home':
    with st.sidebar.expander("About the Gemini & DALL-E2"):
        st.write(
            '''
                **Unleash the power of AI for visual and textual creativity!**

                - **Understand your images with Gemini Pro Vision:** Get detailed textual descriptions of your images.
                - **Bring your words to life:** Generate creative images based on Gemini's description using DALL-E.     
            ''')
    generated_content = None
    image_generation_service = None
    st.sidebar.subheader("Describe and Generate Image!")
    st.sidebar.write(f'You choose to replicate artstyle from: ***{artist_name}***')
    uploaded_file = st.sidebar.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if "artist_name" in st.session_state:
        artist_name = st.session_state["artist_name"]
        if artist_name == None:
            prompt_2 = 'random'
        else:
            prompt_2 = artist_name
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.sidebar.image(img, caption="Uploaded Image", use_column_width=True)
            if st.sidebar.button("Describe Image"):
                with st.spinner("Describing image"):
                    try:
                        generated_content = generate_content(uploaded_file)
                        st.session_state["generated_content"] = generated_content
                    except Exception as e:
                        st.error(f"Error generating description: {e}")
        except Exception as e:
            st.error("Invalid image format. Please upload a PNG, JPG, or JPEG image.")
    if "generated_content" in st.session_state:
        generated_content = st.session_state["generated_content"]

    # Generate image based on description (optional)
    if generated_content is not None:
        st.write(generated_content)
        image_size_option = st.selectbox("Image size", (256, 512, 1024), index=0)

        if st.button("Generate Image"):
            with st.spinner("Generating your art"):
                try:
                    prompt = generated_content + f' in {prompt_2} artstyle'
                    generated_images = generate_dalle(prompt)
                    st.session_state['generated_image_up'] = generated_images
                    st.image(generated_images, caption=f"Image generated using Dall-E 2")
                except Exception as e:
                    st.error(f"Error generating image: {e}")
                       
elif choice == "All Generated Images":
    st.write("All generated images will show up here until close the app.")
    if 'generated_images' not in st.session_state:
        # Initialize an empty list if no images exist yet
        st.session_state['generated_images'] = []
    # Add the newly generated image (if applicable)
    if 'generated_image_up' in st.session_state:
        # Append the image to the list and clear the key afterwards
        st.session_state['generated_images'].append(st.session_state['generated_image_up'])
        del st.session_state['generated_image_up']
    # Display all images stored in the list
    for image in st.session_state['generated_images']:
        st.image(image)
