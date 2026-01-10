import gradio as gr
import requests
import os
import math
from visualizer import get_rice_visualizer_html

# Default backend URL (can be overridden by env var)
DEFAULT_API_URL = os.getenv("API_URL", "https://mlops-bc-latest.onrender.com")

# --- Helper logic for derived features ---
def calculate_equiv_diameter(area):
    """Derived Feature: EquivDiameter = sqrt(4 * Area / pi)"""
    if area <= 0: return 0.0
    return round(math.sqrt(4 * area / math.pi), 2)

def calculate_shape_descriptors(major, minor):
    """
    Derived Features:
    - AspectRatio = Major / Minor
    - Eccentricity = sqrt(1 - (Minor/Major)^2)
    """
    if minor <= 0: return 0.0, 0.0 # Avoid div by zero
    aspect_ratio = major / minor
    
    if major <= 0: return 0.0, round(aspect_ratio, 3)
    
    ratio_sq = (minor / major) ** 2
    eccentricity = 0.0
    if ratio_sq <= 1:
        eccentricity = math.sqrt(1 - ratio_sq)
        
    return round(eccentricity, 4), round(aspect_ratio, 4)

def calculate_roundness(area, perimeter):
    """Derived Feature: Roundness = (4 * pi * Area) / Perimeter^2"""
    if perimeter <= 0: return 0.0
    val = (4 * math.pi * area) / (perimeter ** 2)
    return round(val, 4)

def classify_rice(
    area, major_axis, minor_axis, eccentricity, 
    convex_area, equiv_diameter, extent, perimeter, 
    roundness, aspect_ratio, api_url, model_name="xgboost_binary.onnx"
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
        "AspectRation": float(aspect_ratio),
        "ModelName": str(model_name)
    }
    
    try:
        # Debug print
        print(f"Sending request to {url} using model: {model_name}")
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result.get("predicted_class", "Unknown")
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except ValueError:
        return "Error: Invalid response from server"

def calculate_all_geometry(major, minor, rotation_deg=0):
    """
    Calculates all derived geometric features based on Major and Minor axes
    assuming the grain is roughly elliptical.
    
    Includes rotation to calculate correct Extent (BoundingBox Area).
    """
    if major <= 0 or minor <= 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
    
    # Ellipse approximation
    a = major / 2
    b = minor / 2
    
    # Area (mm^2)
    area = math.pi * a * b
    
    # Perimeter (Ramanujan approximation) (mm)
    h = ((a - b) ** 2) / ((a + b) ** 2)
    perimeter = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    
    # Aspect Ratio
    aspect_ratio = major / minor
    
    # Eccentricity
    eccentricity = 0.0
    if aspect_ratio >= 1:
        eccentricity = math.sqrt(1 - (minor/major)**2)
    else:
        eccentricity = math.sqrt(1 - (major/minor)**2)
        
    # Equiv Diameter (mm)
    equiv_diameter = math.sqrt(4 * area / math.pi)
    
    # Roundness
    roundness = (4 * math.pi * area) / (perimeter ** 2)
    
    # Convex Area (approximate slightly larger than Area for rice grains) (mm^2)
    convex_area = area * 1.02
    
    # Extent (Area / BoundingBox Area)
    # Bounding Box of rotated ellipse:
    theta = math.radians(rotation_deg)
    # Half-width of bbox
    wx = math.sqrt( (a * math.cos(theta))**2 + (b * math.sin(theta))**2 )
    # Half-height of bbox
    wy = math.sqrt( (a * math.sin(theta))**2 + (b * math.cos(theta))**2 )
    bbox_area = (2 * wx) * (2 * wy)
    
    extent = area / bbox_area if bbox_area > 0 else 0
    
    return (
        int(area),
        round(eccentricity, 4),
        int(convex_area),
        round(equiv_diameter, 2),
        round(extent, 4),
        round(perimeter, 2),
        round(roundness, 4),
        round(aspect_ratio, 4)
    )

def build_ui():
    with gr.Blocks(title="Rice Classification UI") as demo:
        gr.Markdown("# 🍚 Rice Classification System")
        gr.Markdown("Enter the geometric features of the rice grain to classify it as **Jazmine** or **Gonen**.")
        
        # Global API URL (applied to both tabs)
        with gr.Row():
            api_url = gr.Textbox(label="Backend API URL", value=DEFAULT_API_URL, scale=3)
            # Dropdown to select model
            model_selector = gr.Dropdown(
                label="Model Selection", 
                choices=["xgboost_binary.onnx", "tabnet_binary.onnx"],
                value="xgboost_binary.onnx",
                scale=1
            )

        with gr.Tabs():
            # =========================================================================
            # TAB 1: SIMULATOR (Simple Mode)
            # =========================================================================
            with gr.TabItem("Simulator (Simple)"):
                gr.Markdown("### 🛠️ Grain Simulator")
                gr.Markdown("Adjust the physical dimensions (Major & Minor Axis) to simulate a rice grain. **Rotation** affects the Extent (Bounding Box).")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        sim_major = gr.Slider(label="Major Axis Length", minimum=70, maximum=190, value=164.16, step=0.1)
                        sim_minor = gr.Slider(label="Minor Axis Length", minimum=30, maximum=90, value=50.17, step=0.1)
                        sim_rotation = gr.Slider(label="Grain Rotation (Degrees)", minimum=0, maximum=90, value=0, step=1)
                    
                    with gr.Column(scale=1):
                        # Visualizer HTML Component
                        vis_html = gr.HTML(get_rice_visualizer_html())
                
                gr.Markdown("#### Calculated Geometry")
                with gr.Group():
                    with gr.Row():
                        sim_area = gr.Number(label="Area", value=0, precision=0)
                        sim_perim = gr.Number(label="Perimeter", value=0, precision=2)
                        sim_equiv = gr.Number(label="Equiv Diameter", value=0, precision=2)
                        sim_convex = gr.Number(label="Convex Area", value=0, precision=0)
                    with gr.Row():
                        sim_ecc = gr.Number(label="Eccentricity", value=0, precision=4)
                        sim_extent = gr.Number(label="Extent", value=0, precision=4)
                        sim_round = gr.Number(label="Roundness", value=0, precision=4)
                        sim_aspect = gr.Number(label="Aspect Ratio", value=0, precision=4)
                
                sim_btn = gr.Button("Classify Simulated Grain", variant="primary")
                sim_output = gr.Textbox(label="Prediction", interactive=False)

                # Event: Update calculated fields when sliders move
                def update_sim_fields(maj, min, rot):
                    return calculate_all_geometry(maj, min, rot)
                
                sim_inputs = [sim_major, sim_minor, sim_rotation]
                sim_display_outputs = [sim_area, sim_ecc, sim_convex, sim_equiv, sim_extent, sim_perim, sim_round, sim_aspect]

                # JavaScript loop to update the p5 visualization
                # This function is executed in the browser when sliders change.
                # It calls the globally exposed method in visualizer.py
                update_viz_js = """
                (maj, min, rot) => {
                    const iframe = document.getElementById('rice-visualizer-iframe');
                    if (iframe && iframe.contentWindow) {
                        iframe.contentWindow.postMessage({
                            type: 'update_params', 
                            major: maj, 
                            minor: min, 
                            rotation: rot
                        }, '*');
                    }
                    return [maj, min, rot];
                }
                """
                
                # Trigger on change (Both Python calculation and JS Visualization)
                sim_major.change(fn=update_sim_fields, inputs=sim_inputs, outputs=sim_display_outputs, js=update_viz_js)
                sim_minor.change(fn=update_sim_fields, inputs=sim_inputs, outputs=sim_display_outputs, js=update_viz_js)
                sim_rotation.change(fn=update_sim_fields, inputs=sim_inputs, outputs=sim_display_outputs, js=update_viz_js)
                
                # Trigger on load
                demo.load(fn=update_sim_fields, inputs=sim_inputs, outputs=sim_display_outputs, js=update_viz_js)

                # Event: Classify
                def classify_simulation(maj, min, rot, url, model):
                    # Recalculate to ensure fresh data
                    (area, ecc, convex, equiv, extent, perim, round_val, aspect) = calculate_all_geometry(maj, min, rot)
                    return classify_rice(
                        area, maj, min, ecc, convex, equiv, extent, perim, round_val, aspect, url, model
                    )
                
                sim_btn.click(
                    fn=classify_simulation,
                    inputs=[sim_major, sim_minor, sim_rotation, api_url, model_selector],
                    outputs=sim_output
                )

            # =========================================================================
            # TAB 2: ADVANCED (Manual Mode)
            # =========================================================================
            with gr.TabItem("Advanced (Manual)"):
                gr.Markdown("### 🎛️ Advanced Controls")
                gr.Markdown("Manually fine-tune every feature. 'Interlocking' logic is enabled to keep derived features consistent.")
                
                with gr.Row():
                    with gr.Column():
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
                        
                        submit_btn = gr.Button("Classify (Advanced)", variant="primary")
                    
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
                
                # --- Interlocking Logic (Reactive Updates) ---
                # 1. Area drives EquivDiameter
                area.change(fn=calculate_equiv_diameter, inputs=[area], outputs=[equiv_diameter])
                
                # 2. Area + Perimeter drives Roundness
                area.change(fn=calculate_roundness, inputs=[area, perimeter], outputs=[roundness])
                perimeter.change(fn=calculate_roundness, inputs=[area, perimeter], outputs=[roundness])
                
                # 3. Major + Minor drives Eccentricity & AspectRatio
                major_axis.change(fn=calculate_shape_descriptors, inputs=[major_axis, minor_axis], outputs=[eccentricity, aspect_ratio])
                minor_axis.change(fn=calculate_shape_descriptors, inputs=[major_axis, minor_axis], outputs=[eccentricity, aspect_ratio])

                submit_btn.click(
                    fn=classify_rice,
                    inputs=[
                        area, major_axis, minor_axis, eccentricity, 
                        convex_area, equiv_diameter, extent, perimeter, 
                        roundness, aspect_ratio, api_url, model_selector
                    ],
                    outputs=output_text
                )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)