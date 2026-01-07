import gradio as gr
import requests
import os

# Default backend URL (can be overridden by env var)
DEFAULT_API_URL = os.getenv("API_URL", "https://mlops-bc-latest.onrender.com")

def classify_rice(
    area, major_axis, minor_axis, eccentricity, 
    convex_area, equiv_diameter, extent, perimeter, 
    roundness, aspect_ratio, api_url
):
    """
    Sends the features to the backend API for classification.
    """
    url = api_url.rstrip("/") + "/classify/"
    
    payload = {
        "Area": int(area),
        "MajorAxisLength": float(major_axis),
        "MinorAxisLength": float(minor_axis),
        "Eccentricity": float(eccentricity),
        "ConvexArea": int(convex_area),
        "EquivDiameter": float(equiv_diameter),
        "Extent": float(extent),
        "Perimeter": float(perimeter),
        "Roundness": float(roundness),
        "AspectRation": float(aspect_ratio)
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result.get("predicted_class", "Unknown")
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except ValueError:
        return "Error: Invalid response from server"

def build_ui():
    with gr.Blocks(title="Rice Classification UI") as demo:
        gr.Markdown("# 🍚 Rice Classification System")
        gr.Markdown("Enter the geometric features of the rice grain to classify it as **Cammeo** or **Osmancik**.")
        
        with gr.Row():
            with gr.Column():
                api_url = gr.Textbox(label="Backend API URL", value=DEFAULT_API_URL)
                
                gr.Markdown("### Geometric Features")
                with gr.Row():
                    area = gr.Slider(label="Area", minimum=2500, maximum=10500, value=6322, step=1)
                    convex_area = gr.Slider(label="Convex Area", minimum=2500, maximum=11500, value=6525, step=1)
                
                with gr.Row():
                    major_axis = gr.Slider(label="Major Axis Length", minimum=70, maximum=190, value=164.16, step=0.01)
                    minor_axis = gr.Slider(label="Minor Axis Length", minimum=30, maximum=90, value=50.17, step=0.01)
                
                with gr.Row():
                    perimeter = gr.Slider(label="Perimeter", minimum=190, maximum=520, value=359.51, step=0.01)
                    equiv_diameter = gr.Slider(label="Equiv Diameter", minimum=50, maximum=120, value=89.71, step=0.01)
                
                with gr.Row():
                    eccentricity = gr.Slider(label="Eccentricity", minimum=0.6, maximum=1.0, value=0.95, step=0.001)
                    extent = gr.Slider(label="Extent", minimum=0.3, maximum=0.9, value=0.56, step=0.001)
                
                with gr.Row():
                    roundness = gr.Slider(label="Roundness", minimum=0.1, maximum=1.0, value=0.61, step=0.001)
                    aspect_ratio = gr.Slider(label="Aspect Ratio", minimum=1.0, maximum=4.0, value=3.27, step=0.001)
                
                submit_btn = gr.Button("Classify", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Prediction Result")
                output_text = gr.Textbox(label="Predicted Class", interactive=False)
                
                gr.Markdown("""
                ### Feature Guide
                * **Area**: Area of the rice grain in pixels.
                * **Major/Minor Axis**: Length of the main axes of the grain.
                * **Eccentricity**: Measure of how much the grain deviates from a circle.
                * **Convex Area**: Area of the smallest convex polygon enclosing the grain.
                * **Extent**: Ratio of the area to the bounding box area.
                * **Perimeter**: Circumference of the grain.
                * **Roundness**: Measure of how circular the grain is.
                * **Aspect Ratio**: Ratio of Major Axis to Minor Axis.
                """)

        submit_btn.click(
            fn=classify_rice,
            inputs=[
                area, major_axis, minor_axis, eccentricity, 
                convex_area, equiv_diameter, extent, perimeter, 
                roundness, aspect_ratio, api_url
            ],
            outputs=output_text
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)