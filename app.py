import warnings
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import re
import inflect  # <-- NEW: For morphological singularization

# Removed warnings.filterwarnings for torch.classes so errors are visible
# warnings.filterwarnings("ignore", message=r".*torch\.classes.*")
# warnings.filterwarnings("ignore", category=UserWarning)

# Set page configuration
st.set_page_config(page_title="Skin Cancer Analyzer", layout="wide")

# Initialize inflect engine for singularization
p = inflect.engine()

def load_model():
    """Load the model and image processor from HuggingFace"""
    try:
        model_name = "Anwarkh1/Skin_Cancer-Image_Classification"
        
        # Use fast image processor
        image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        return model, image_processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.warning("Please check your internet connection and try restarting the application.")
        return None, None

def preprocess_image(image, image_processor):
    """Preprocess image for model input"""
    try:
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preprocess the image
        inputs = image_processor(images=image, return_tensors="pt")
        return inputs
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict(model, inputs):
    """Make predictions using the model"""
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        probabilities = probabilities.numpy()
        
        # Get class names and create a dictionary
        class_names = model.config.id2label
        predictions = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        # Sort by probability in descending order
        sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_predictions
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def plot_prediction_results(predictions, threshold=0.01):
    """Plot the prediction results with professional matplotlib styling"""
    try:
        plt.style.use('ggplot')
        
        filtered_predictions = {k: v for k, v in predictions.items() if v >= threshold}
        if not filtered_predictions:
            filtered_predictions = predictions
        
        top_class = max(predictions.items(), key=lambda x: x[1])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(filtered_predictions.keys())
        probs = list(filtered_predictions.values())
        
        # Sort by probability
        sorted_data = sorted(zip(labels, probs), key=lambda x: x[1])
        labels, probs = zip(*sorted_data)
        
        if len(probs) > 1:
            norm_values = np.array(probs)
            norm_values = (norm_values - min(norm_values)) / (max(norm_values) - min(norm_values) or 1)
            norm_values = np.power(norm_values, 0.7)
            colors = plt.cm.viridis(norm_values)
        else:
            colors = plt.cm.viridis(np.array([0.8]))
        
        bars = ax.barh(labels, probs, color=colors, height=0.6, 
                       edgecolor='black', linewidth=0.5, alpha=0.9)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            for dx, dy in [(-0.5, -0.5), (0.5, 0.5)]:
                ax.text(width + 0.01 + dx*0.001, bar.get_y() + bar.get_height()/2 + dy*0.001, 
                        f"{probs[i]:.4f}", va='center', ha='left', 
                        color='lightgray', alpha=0.3, fontweight='bold', fontsize=9)
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{probs[i]:.4f}", va='center', ha='left', 
                    color='black', fontweight='bold', fontsize=9)
            
            # If not top element, show difference from top
            if i < len(probs) - 1 and probs[-1] > 0:
                diff_pct = ((probs[-1] - probs[i]) / probs[-1]) * 100
                ax.text(width + 0.15, bar.get_y() + bar.get_height()/2, 
                        f"({diff_pct:.1f}% less)", va='center', ha='left', 
                        color='#555555', fontsize=8, style='italic')
        
        ax.set_xlabel('Probability', fontsize=11, fontweight='bold')
        ax.set_title(f"Diagnosis: {top_class[0]} (Confidence: {top_class[1]:.4f})", 
                     fontsize=14, fontweight='bold', pad=15)
        
        ax.set_xlim(0, 1.15)
        ax.grid(axis='x', linestyle='--', alpha=0.4, which='both')
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.05, 0.05), minor=True)
        ax.grid(axis='x', linestyle=':', alpha=0.2, which='minor')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='x', which='minor', bottom=True)
        
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f0f0f5')
        
        # Optional inset for close predictions
        if len(probs) > 1:
            top_values = sorted(probs, reverse=True)[:min(3, len(probs))]
            min_zoom = max(0, min(top_values) - 0.05)
            max_zoom = min(1, max(top_values) + 0.05)
            
            if max_zoom - min_zoom < 0.3:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                axins = inset_axes(ax, width="40%", height="30%", loc="center right")
                
                top_indices = [i for i, p in enumerate(probs) if p >= min_zoom]
                inset_labels = [labels[i] for i in top_indices]
                inset_probs = [probs[i] for i in top_indices]
                inset_colors = [colors[i] for i in top_indices]
                
                inset_bars = axins.barh(inset_labels, inset_probs, color=inset_colors,
                                        height=0.6, edgecolor='black', linewidth=0.5)
                
                for i, bar in enumerate(inset_bars):
                    width = bar.get_width()
                    text_color = 'black' if width < 0.5 else 'white'
                    axins.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                               f"{inset_probs[i]:.4f}", va='center', ha='left', 
                               color=text_color, fontweight='bold', fontsize=8)
                
                axins.set_xlim(min_zoom, max_zoom)
                axins.spines['top'].set_visible(False)
                axins.spines['right'].set_visible(False)
                axins.set_title("Zoomed View", fontsize=10)
                axins.grid(axis='x', linestyle='--', alpha=0.3)
                axins.tick_params(axis='both', labelsize=8)
                
                from matplotlib.patches import Rectangle
                rec = Rectangle((0, 0), 1, 1, transform=axins.transAxes,
                                fill=False, edgecolor='gray', linewidth=1)
                axins.add_patch(rec)
                
                from mpl_toolkits.axes_grid1.inset_locator import mark_inset
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", alpha=0.5)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting results: {str(e)}")
        return None

def main():
    """Main function for the Streamlit application"""
    # Page header
    st.title("🔬 Skin Cancer Analyzer")
    st.markdown("Upload an image of a skin lesion for analysis")
    
    # Medical disclaimer
    st.warning(
        "**IMPORTANT DISCLAIMER**: This application is for educational purposes only. "
        "It should not be used for diagnostic purposes. Always consult with a qualified "
        "healthcare professional for medical advice."
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, image_processor = load_model()
    
    if model is None or image_processor is None:
        st.error("Failed to load model. Please reload the page and try again.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Analyzing image..."):
                inputs = preprocess_image(image, image_processor)
                
                if inputs is not None:
                    predictions = predict(model, inputs)
                
                    if predictions is not None:
                        with col2:
                            st.subheader("Analysis Results")
                            
                            fig = plot_prediction_results(predictions)
                            if fig:
                                st.pyplot(fig)
                            
                            top_class = max(predictions.items(), key=lambda x: x[1])
                            st.success(f"Predicted Class: **{top_class[0]}**")
                            st.info(f"Confidence: **{top_class[1]:.4f}**")
                            
                            st.subheader("Detailed Probabilities")
                            prediction_df = {
                                "Class": list(predictions.keys()),
                                "Probability": list(predictions.values())
                            }
                            st.dataframe(prediction_df)
                            
                            st.subheader("About the Skin Condition")
                            
                            CONDITION_MAP = {
                                "basal cell carcinoma": [
                                    "bcc", "basal cell carcinoma", "basal_cell_carcinoma",
                                    "basal-cell-carcinoma", "basalcellcarcinoma"
                                ],
                                "squamous cell carcinoma": [
                                    "scc", "squamous cell carcinoma", "squamous_cell_carcinoma",
                                    "squamous-cell-carcinoma"
                                ],
                                "melanocytic nevus": [
                                    "nevus", "melanocytic nevus", "melanocytic_nevus",
                                    "common nevus", "common_nevus"
                                    # We don't need "melanocytic nevi" here because
                                    # morphological singularization now covers it.
                                ],
                                "melanoma": [
                                    "melanoma", "malignant melanoma", "malignant_melanoma"
                                ],
                                "actinic keratosis": [
                                    "actinic keratosis", "solar keratosis", "actinic_keratosis","actinic keratoses"
                                ],
                                "seborrheic keratosis": [
                                    "seborrheic keratosis", "seborrheic_keratosis",
                                    "seborrheic-keratosis", "seborrheic verruca"
                                ],
                                "dermatofibroma": [
                                    "dermatofibroma", "dermatofibroma_benign"
                                ],
                                "vascular lesion": [
                                    "vascular lesion", "vascular_lesion", "angioma"
                                ],
                                "pigmented benign keratosis": [
                                    "pigmented benign keratosis", "pigmented_benign_keratosis",
                                    "benign keratosis"
                                ]
                            }
                            
                            condition_info = {
                                "actinic keratosis": """**Description**: A rough, scaly patch on skin that develops from years of sun exposure.

**Key Characteristics**:
• Rough, dry, scaly patches
• Size typically 2-6mm
• Color ranges from pink to red to brown

**Common Locations**:
• Face, lips, ears
• Neck, forearms, hands
• Bald scalp in men

**Risk Factors**:
• Fair skin
• Age over 40
• Cumulative sun exposure
• Weakened immune system

**Important**: Considered pre-cancerous; can develop into squamous cell carcinoma if untreated.""",
    
                                "basal cell carcinoma": """**Description**: The most common type of skin cancer, highly treatable when caught early.

**Key Characteristics**:
• Pearly, waxy bump
• Flat, flesh-colored or brown scar-like lesion
• Bleeding or scabbing sore that heals and returns

**Common Locations**:
• Face, particularly nose
• Sun-exposed areas
• Can occur anywhere

**Risk Factors**:
• Chronic sun exposure
• Fair skin
• Radiation therapy
• Family history

**Important**: Slow-growing but can be locally destructive if left untreated.""",
    
                                "melanoma": """**Description**: The most dangerous form of skin cancer, can spread to other parts of the body.

**Key Characteristics (ABCDE rule)**:
• Asymmetry
• Border irregularity
• Color variation
• Diameter > 6mm
• Evolution/change over time

**Common Locations**:
• Any part of body
• Men: often trunk
• Women: often legs

**Risk Factors**:
• UV exposure
• Multiple moles
• Fair skin
• Family history

**Important**: Early detection crucial for successful treatment.""",
    
                                "nevus": """**Description**: Common moles, usually harmless pigmented growths.

**Key Characteristics**:
• Round or oval shape
• Even coloring
• Smooth surface
• < 6mm diameter

**Types**:
• Congenital (present at birth)
• Acquired (develop over time)
• Dysplastic (atypical appearance)

**Monitoring**:
• Track changes in size, shape, color
• Note any new symptoms
• Regular skin checks recommended

**Important**: Most are benign but should be monitored for changes.""",
    
                                "seborrheic keratosis": """**Description**: Common benign skin growths that often appear with age.

**Key Characteristics**:
• Waxy, scaly, slightly raised
• Light brown to black
• "Stuck-on" appearance
• Variable size

**Common Locations**:
• Face
• Chest
• Back
• Abdomen

**Facts**:
• Not pre-cancerous
• Multiple can appear over time
• Generally harmless
• No treatment needed unless irritated

**Important**: While benign, any rapid changes should be evaluated.""",
    
                                "squamous cell carcinoma": """**Description**: Second most common skin cancer, develops in squamous cells.

**Key Characteristics**:
• Firm, red nodules
• Flat lesions with scaly surface
• May have raised edges
• Can be tender to touch

**Common Locations**:
• Sun-exposed areas
• Face, ears, lips
• Back of hands
• Scalp

**Risk Factors**:
• Extensive sun exposure
• Fair skin
• Age over 50
• Male gender

**Important**: More aggressive than basal cell carcinoma; can spread if untreated.""",
    
                                "vascular lesion": """**Description**: Abnormalities of blood vessels visible on the skin.

**Types**:
• Cherry angiomas
• Spider angiomas
• Port wine stains
• Hemangiomas

**Characteristics**:
• Red to purple color
• May be flat or raised
• Size varies greatly
• Can blanch with pressure

**Risk Factors**:
• Age
• Pregnancy
• Liver disease
• Genetic factors

**Important**: Most are benign but evaluation recommended for new or changing lesions.""",
    
                                "dermatofibroma": """**Description**: A common benign fibrous nodule usually found on the limbs.

**Key Characteristics**:
• Small, firm bump
• Brown to reddish-brown color
• May dimple when pinched
• Usually less than 1cm in diameter

**Common Locations**:
• Legs
• Arms
• Trunk (less common)

**Facts**:
• Often appears after minor trauma
• Generally harmless and requires no treatment
• Can persist indefinitely
• May be itchy or tender

**Important**: Rarely transforms into malignancy; removal only needed if symptomatic.""",
    
                                "pigmented benign keratosis": """**Description**: A benign growth of the outer layer of the skin.

**Key Characteristics**:
• Well-defined, round or oval shaped
• Light brown to dark brown color
• Flat or slightly raised
• Smooth or slightly rough surface

**Common Locations**:
• Face
• Back
• Chest
• Extremities

**Facts**:
• Not cancerous
• Related to sun exposure
• More common in adults
• May resemble melanoma but is harmless

**Important**: Should be monitored for changes in appearance but generally requires no treatment."""
                            }
                            
                            # --- NEW: updated morphological singularization in this function ---
                            def normalize_condition_name(name: str) -> str:
                                """Normalize name to handle underscores/dashes and convert plurals to singular."""
                                name = name.lower().replace('_', ' ')
                                clean = re.sub(r'[_-]', ' ', name.strip())
                                words = clean.split()

                                singular_words = []
                                for w in words:
                                    # Attempt singular form
                                    if w == "keratosi":
                                        singular_form = "keratosis"
                                    elif w == "nevu":
                                        singular_form = "nevus"
                                    else:
                                        singular_form = p.singular_noun(w)
                                    singular_words.append(singular_form if singular_form else w)

                                return ' '.join(singular_words)

                            def find_matching_condition(detected_condition, condition_info, condition_map):
                                normalized_detected = normalize_condition_name(detected_condition)
                                print("Debug: Attempting to match condition:", normalized_detected)

                                # Word set match (more robust)
                                detected_words = set(normalized_detected.split())
                                for condition in condition_info.keys():
                                    normalized_condition = normalize_condition_name(condition)
                                    print(f"Debug: Comparing '{normalized_detected}' with '{normalized_condition}'")
                                    condition_words = set(normalized_condition.split())
                                    if detected_words == condition_words:
                                        print("Debug: Found word set match:", condition)
                                        return condition

                                # Combined direct and variant matching
                                for condition, variants in list(condition_map.items()) + list(condition_info.items()):
                                    normalized_condition = normalize_condition_name(condition)
                                    print(f"Debug: Comparing '{normalized_detected}' with '{normalized_condition}'")
                                    if normalized_detected == normalized_condition:
                                        print("Debug: Found direct or variant match:", condition)
                                        return condition

                                    if isinstance(variants, list):
                                        normalized_variants = [normalize_condition_name(v) for v in variants]
                                        if normalized_detected in normalized_variants:
                                            print("Debug: Found variant match:", condition)
                                            return condition

                                # Substring match as a last resort
                                for condition in condition_info.keys():
                                    if normalized_detected in normalize_condition_name(condition):
                                        print("Debug: Found substring match:", condition)
                                        return condition

                                print("Debug: No match found for:", normalized_detected)
                                print("Debug: Available conditions:", list(condition_info.keys()))
                                return None

                            detected_condition = top_class[0]

                            try:
                                matching_condition = find_matching_condition(detected_condition, condition_info, CONDITION_MAP)
                                if matching_condition and matching_condition in condition_info:
                                    st.markdown(condition_info[matching_condition])
                                else:
                                    st.warning(f"Detailed information for '{detected_condition}' is not yet available in our database.")
                                    st.info("Please consult a healthcare professional for accurate medical advice.")
                            except Exception as e:
                                st.error(f"Error matching condition: {str(e)}")
                                st.info("Please consult a healthcare professional for accurate medical advice.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.warning("Please try another image or restart the application.")

if __name__ == "__main__":
    main()
