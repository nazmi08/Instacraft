## Instacraft: Unleash Your Creativity with AI-Powered Image Transformation

This repository houses **Instacraft**, a Streamlit application that empowers users to **transform ordinary images into artistic masterpieces**, inspired by the works of renowned artists like Vincent van Gogh, Claude Monet, and more. 

### Harnessing the Power of AI

Instacraft leverages a powerful combination of cutting-edge AI models:

* **Gemini Pro Vision:** This state-of-the-art model analyzes and describes the content of your image, generating a detailed textual representation.
* **DALL-E 2:** Utilizing the generated description, DALL-E 2 creates a new image, replicating the style and essence of your chosen artistic inspiration.
* **Stable Diffusion v2 and XL:** These free-to-use models from Stability AI offer additional options for generating high-quality, artistic image transformations.

This synergistic combination unlocks a unique avenue for creative exploration, enabling you to:

* **Derive rich descriptions from existing images.**
* **Spark the creation of entirely new artworks using AI-powered artistic styles.**

### Getting Started

**1. Installation:**

* **Clone the repository:**

```
git clone https://github.com/your-username/Instacraft.git
```

* **Create a virtual environment (recommended):**

```
python -m venv venv
source venv/bin/activate
```

* **Install dependencies:**

```
pip install -r requirements.txt
```

**2. Running the App:**

**Important Note:**

* **Hardware Requirements:** Running the `Original_App_RunLocally.py` script requires a powerful graphics card (GPU) with sufficient memory to process the AI models effectively. We recommend a minimum configuration of **NVIDIA GTX 1650 with 4GB of dedicated RAM, 32 GB of system RAM, and a processor like AMD Ryzen 5 600H or equivalent**. 

**Without these hardware specifications, you may encounter issues like "CUDA out of memory" errors.**

**2.1 Locally:**

```
streamlit run Original_App_RunLocally.py
```

* **Open your web browser and navigate to http://localhost:8501**

**2.2 Lite Version:**

For a simplified experience, access the lite version directly through your browser at: Instacraft: [https://instacraft.streamlit.app/](https://instacraft.streamlit.app/)

**3. Unleashing Your Creativity with Instacraft:**

1. Launch the application using the instructions above.
2. Select your desired artistic style and upload an image you wish to transform.
3. Instacraft will analyze your image and generate a textual description.
4. Further by generate your image and let Instacraft work its magic!

Instacraft empowers you to explore the boundless possibilities of AI-driven artistic creation. We invite you to experiment, unleash your creativity, and discover the transformative power of AI in the world of art.

### Contributing

We welcome contributions from the developer community! 
