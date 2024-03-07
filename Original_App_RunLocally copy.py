from PIL import Image
import openai
import torch

# Diffusers library imports
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForImage2Image,
)

# Google AI API import
import google.generativeai as genai

# Streamlit framework import
import streamlit as st

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

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.pexels.com/photos/1029618/pexels-photo-1029618.jpeg");
background-size: cover;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}

[data-testid="stSidebarContent"] {
background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Monet_-_Impression%2C_Sunrise.jpg/1280px-Monet_-_Impression%2C_Sunrise.jpg");
background-position: center;
}
</style>
"""

st.set_page_config(
    page_title = 'Reimagine AI',
    page_icon = '🖼️'
)

# st.markdown(page_bg_img, unsafe_allow_html=True)

def generate_dalle(msg, image_size=256, artist="None"):
    if artist == "None":
        prompt=f'{msg} in random style'
    else:
        prompt=f'{msg} in {artist} artstyle'
    response = openai.images.generate(
        model='dall-e-2',
        prompt=prompt,
        n=1,
        size=f'{image_size}x{image_size}',
        quality='standard'
    )
    image_url = response.data[0].url
    return image_url

def generate_image_using_diffusers(input_text, num_images=1,image_size=512, artist='None'):
    try:
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing("max")
        # Define the prompt and generate the image
        if artist == 'None':
            prompt = input_text
        else:
            prompt = input_text + f' in {artist} artstyle'
        image = pipe(prompt, height=image_size, width=image_size, num_images_per_prompt=num_images, num_inference_steps=50).images
        torch.cuda.empty_cache()
        return image
    except Exception as e:
        print(f"Error generation image: {e}")
        raise e # Re-raise the exception for Streamlit handling


def generate_image_using_diffusers_img2img(uploaded_file, num_images=1, image_size=512, artist='None'):
    try:
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing("max")
        init_image = Image.open(uploaded_file).convert("RGB")
        # Define the prompt and generate the image
        if artist == 'None':
            prompt = 'random style'
        else:
            prompt = f"in {artist} artstyle"
        image = pipe(prompt=prompt, image=init_image, strength=0.5, guidance_scale=10.5, height=image_size, width=image_size, num_images_per_prompt=num_images, num_inference_steps=50).images
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        raise e  # Re-raise the exception for Streamlit handling

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

def generate_image_sdxl(uploaded_file, num_images=1, image_size=512, artist='None'):
    try:
        pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, variant="fp16", use_safetensors=True)
        pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image)
        pipeline.enable_sequential_cpu_offload()
        pipeline.enable_attention_slicing("max")
        init_image = Image.open(uploaded_file).convert("RGB")
        prompt = f"in {artist} artstyle"
        image = pipeline(prompt, image=init_image, strength=0.5, guidance_scale=10.5, num_images_per_prompt=num_images, height=image_size, width=image_size, num_inference_steps=50).images
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        raise e # Re-raise the exception for Streamlit handling
    
def generate_image_text_sdxl(input_text, num_images=1, image_size=512, artist='None'):
    try:
        model_2 = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = AutoPipelineForText2Image.from_pretrained(model_2, torch_dtype=torch.float32, variant="fp16", use_safetensors=True)
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing("max")
        prompt = input_text
        prompt_2 = f'in {artist} style'
        image = pipe(prompt=prompt, prompt_2=prompt_2, num_images_per_prompt=num_images, height=image_size, width=image_size, num_inference_steps=50).images
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        raise e # Re-raise the exception for Streamlit handling
    
def display_artist_images(artist_name, image_urls):
  if artist_name != None:
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, image_url in enumerate(image_urls):
        try:
            col = [col1, col2, col3, col4, col5][i % 5]
            col.image(image_url, use_column_width=True, caption=f"Image {i + 1}")
        except Exception as e:
            st.error(f"Error loading image: {e}")

choice = st.sidebar.selectbox("Select your choice", ["Home", "DALL-E", "Gemini", "SDv2", "SDXL", "All Generated Images"])

if choice == "Home":
    st.image("https://i.imgur.com/RPQdYtB.jpeg")
    st.title("InstaCraft - Unleash Your Creative Vision")
    with st.expander("**About the app**"):
        st.write(    
    """
        **InstaCraft** is AI Art Generator with powerful tool to **transform your ideas into stunning visuals.** 
        It utilizes cutting-edge AI to **instantly apply diverse artistic styles** to your images, empowering you to 
        **explore creative possibilities** and **visualize your concepts** in an engaging and impactful way.\n
        **Benefits:**\n
        - **Spark creative inspiration:** Generate unique image variations based on your chosen artistic style.
        - **Enhance individual expression:** Personalize your images with an artistic flair.
        - **Elevate marketing materials:** Capture attention and create visually compelling marketing materials.
        - **Showcase tourist attractions:** Generate visually striking representations of tourist destinations.
    """)
        
    artist_name = st.selectbox("Pick your artist for the image generated artstyle:", tuple(artist_images.keys()), index=0)
    st.session_state["artist_name"] = artist_name
    if artist_name in artist_images:
        display_artist_images(artist_name, artist_images[artist_name])
    else:
        st.info("Artist not found.")
        
elif choice == "DALL-E":
    with st.sidebar.expander("About the DALL-E", expanded=True):
        st.write(
            '''
                **Turn your words into images with Dall-E 2!**
                - **Bring your words to life:** Generate creative images from your text descriptions. 
                - Choose an artistic style (available under "Home") and describe your desired image, like "Petronas Twin Towers with a blue sky background" and it will automatically add the artist's artstyle.      
            ''')
    st.subheader("Image Generation using DALL-E")
    input_text=st.text_input('Enter your prompt: ')
    image_size_option = st.selectbox("Image size", (256, 512, 1024), index=0)
    if "artist_name" in st.session_state:
        artist_name = st.session_state["artist_name"]
        st.write(f'Artstyle chosen is from: ***{artist_name}***')
        prompt_2 = artist_name
    prompt = input_text + f'in {prompt_2} artstyle'
    if prompt is not None:
        if st.button("Generate Image"):
            with st.spinner("Generating your art"):
                img_url=generate_dalle(prompt, image_size=image_size_option, artist=artist_name)
                st.session_state['generated_image_up'] = img_url
                st.image(img_url, caption="Image generated using DALL-E 2")

elif choice == 'Gemini':
    with st.sidebar.expander("About the Gemini", expanded=True):
        st.write(
            '''
                **Unleash the power of AI for visual and textual creativity!**

                - **Understand your images with Gemini Pro Vision:** Get detailed textual descriptions of your images.
                - **Bring your words to life:** Generate creative images based on Gemini's description using DALL-E, Stable Diffusion v2, or Stable Diffusion XL.
                  - **Explore artistic styles:** Choose artstyle from the "Home" section to enhance your image creations.       
            ''')
    generated_content = None
    image_generation_service = None
    st.subheader("Image Description using Gemini")
    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if "artist_name" in st.session_state:
        artist_name = st.session_state["artist_name"]
        st.write(f'Artstyle chosen is from: ***{artist_name}***')
        if artist_name == None:
            prompt_2 = 'random'
        else:
            prompt_2 = artist_name
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True, caption="Uploaded Image")
            if st.button("Describe Image"):
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
        image_generation_service = st.selectbox(
            "Select an image generation service:",
            ("DALL-E 2", "Diffusers (SDv2)", "Diffusers (SDXL)", "None")
        )
        if image_generation_service != "None":
            if image_generation_service == "Diffusers (SDv2)":
                num_images = st.number_input("Number of images (1-4)", min_value=1, max_value=4, value=1)
                image_size_option = st.selectbox("Image size", (512, 768, 1024), index=0)

            if image_generation_service == "Diffusers (SDXL)":
                image_size_option = st.selectbox("Image size", (512, 768, 1024), index = 0)

            if st.button("Generate Image"):
                with st.spinner("Generating your art"):
                    try:
                        prompt = generated_content + f' in {prompt_2} artstyle'
                        if image_generation_service == "Diffusers (SDv2)":
                            generated_images = generate_image_using_diffusers(prompt, num_images=num_images, image_size=image_size_option)
                            for i, image in enumerate(generated_images):
                                st.session_state['generated_image_up'] = image
                                st.image(image, caption=f"Image {i+1} generated using Gemini Pro Vision + Huggingface Diffusers (SDv2)")
                        elif image_generation_service == "Diffusers (SDXL)":
                            generated_image = generate_image_text_sdxl(prompt, image_size=image_size_option)
                            st.session_state['generated_image_up'] = generated_image
                            st.image(generated_image, caption=f"Image generated using Gemini Pro Vision + Huggingface Diffusers (SDXL)")
                        if image_generation_service == "DALL-E 2":
                            generated_images = generate_dalle(prompt)
                            st.session_state['generated_image_up'] = generated_images
                            st.image(generated_images, caption=f"Image generated using Dall-E 2")
                    except Exception as e:
                        st.error(f"Error generating image: {e}")

elif choice == "SDv2":
    with st.sidebar.expander("About the SDv2", expanded=True):
        st.write(
        '''
            **Unleash the power of AI for visual creativity!**
            - **Bring your words to life:** Describe your dream image and let Stable Diffusion v2 generate it.
            - **Upload your image:** Generate creative images based on Stable Diffusion v2.
              - **Explore artistic styles:** Choose artstyle from the "Home" section to enhance your image creations.       
        ''')
    choice = st.radio("Choose generation method:", ("Text Prompt", "Image Upload"))
    if "artist_name" in st.session_state:
        artist_name = st.session_state["artist_name"]
        st.write(f'Artstyle chosen is from: ***{artist_name}***')
        prompt_2 = artist_name
    if choice == "Text Prompt":
        st.subheader("Image Generation using Diffusers SDv2")
        input_text = st.text_input("Enter your text")
        num_images = st.number_input("Number of images (1-4)", min_value=1, max_value=4, value=1)
        image_size_option = st.selectbox("Image size", (512, 768, 1024), index=0)
        if input_text is not None:
            # Generate and display transformed image
            if st.button("Generate Image"):
                with st.spinner("Generating your art"):
                    try:
                        generated_images = generate_image_using_diffusers(input_text, num_images=num_images, image_size=image_size_option, artist=artist_name)
                        for i, image in enumerate(generated_images):
                            st.session_state['generated_image_up'] = image
                            st.image(image, caption=f"Image {i+1} generated using Huggingface Diffusers (SDv2)")
                    except Exception as e:
                        st.error(f"Error generating image: {e}")

    elif choice == "Image Upload":
        st.subheader("Image Generation using Diffusers SDv2")
        uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, use_column_width=True, caption="Uploaded Image")
                num_images = st.number_input("Number of images (1-4)", min_value=1, max_value=4, value=1)
                image_size_option = st.selectbox("Image size", (512, 768, 1024), index=0)
                # Generate and display transformed image
                if st.button("Generate Image"):
                    with st.spinner("Generating your art"):
                        try:
                            generated_images = generate_image_using_diffusers_img2img(uploaded_file, num_images=num_images, image_size=image_size_option, artist=artist_name)
                            for i, image in enumerate(generated_images):
                                st.session_state['generated_image_up'] = image
                                st.image(image, caption=f"Image {i+1} generated using Huggingface Diffusers (SDv2)")
                        except Exception as e:
                            st.error(f"Error generating image: {e}")
            except Exception as e:
                st.error("Invalid image format. Please upload a PNG, JPG, or JPEG image.")

elif choice == 'SDXL':
    with st.sidebar.expander("About the SDXL", expanded=True):
        st.write("""
            **Unleash the power of AI for visual creativity!**

            - **Bring your words to life:** Describe your dream image and let **Stable Diffusion XL, the most powerful Stable Diffusion release yet**, generate it with stunning detail and fidelity.
            - **Elevate your existing images:** Upload your photo and let Stable Diffusion XL **transform it into a masterpiece** with unparalleled artistic flair.
            - **Explore artistic styles:** Choose a style from "Home" to further customize your creations and unlock unique visual possibilities.

            **Note:** Due to its enhanced capabilities, Stable Diffusion XL requires longer processing times to generate an image. For example, generating an image at 1024x1024 resolution might take approximately 10 minutes, depending on your hardware specifications.
        """)

    choice = st.radio("Choose generation method:", ("Text Prompt", "Image Upload"))
    if "artist_name" in st.session_state:
        artist_name = st.session_state["artist_name"]
        st.write(f'Artstyle chosen is from: ***{artist_name}***')
        if artist_name == "None":
            prompt_2 = 'random'
        else:
            prompt_2 = artist_name
    if choice == "Text Prompt":
        st.subheader("Image Generation using Diffusers SDXL")
        input_text = st.text_input("Enter your text")
        image_size_option = st.selectbox("Image size", (512, 768, 1024), index=0)
        if input_text is not None:
            # Generate and display transformed image
            if st.button("Generate Image"):
                with st.spinner("Generating your art"):
                    try:
                        prompt = input_text + f'in {prompt_2} artstyle'
                        generated_image = generate_image_text_sdxl(prompt, image_size=image_size_option)
                        st.session_state['generated_image_up'] = generated_image
                        st.image(generated_image, caption=f"Image generated using Huggingface Diffusers (SDXL)")
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
    elif choice == "Image Upload":
        st.subheader("Image Generation using Diffusers SDXL")
        uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png","jpg","jpeg"])

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, use_column_width=True, caption = "Uploaded Image")
                image_size_option = st.selectbox("Image size", (512, 768, 1024), index=0)
                # Generate and display transformed image
                if st.button("Transform Image"):
                    with st.spinner("Transforming image"): 
                        try:
                            generated_image = generate_image_sdxl(uploaded_file, image_size=image_size_option, artist=artist_name)
                            st.session_state['generated_image_up'] = generated_image
                            st.image(generated_image, caption=f"Image generated using Huggingface Diffusers (SDXL)")
                        except Exception as e:
                            st.error(f"Error generating image: {e}")
            except Exception as e:
                st.error("Invalid image format. Please upload a PNG, JPG, or JPEG image.")

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

############### 