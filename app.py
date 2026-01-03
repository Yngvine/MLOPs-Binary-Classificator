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
                    area = gr.Number(label="Area", value=6322)
                    convex_area = gr.Number(label="Convex Area", value=6525)
                
                with gr.Row():
                    major_axis = gr.Number(label="Major Axis Length", value=164.16)
                    minor_axis = gr.Number(label="Minor Axis Length", value=50.17)
                
                with gr.Row():
                    perimeter = gr.Number(label="Perimeter", value=359.51)
                    equiv_diameter = gr.Number(label="Equiv Diameter", value=89.71)
                
                with gr.Row():
                    eccentricity = gr.Number(label="Eccentricity", value=0.95)
                    extent = gr.Number(label="Extent", value=0.56)
                
                with gr.Row():
                    roundness = gr.Number(label="Roundness", value=0.61)
                    aspect_ratio = gr.Number(label="Aspect Ratio", value=3.27)
                
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