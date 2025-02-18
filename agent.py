from agents import Agent
from agents.tools import ToolContext
from google import genai
from google.genai import types
from google.cloud import storage
from typing import Dict, List

# Mock Data and Stubbed Functions (Replace with real implementations)
SAMPLE_IMAGE_URLS = ["gs://site_comparisons/beach_image.png"]
SAMPLE_SOCIAL_MEDIA_POSTS = [
    {"sentiment": "negative", "text": "Concerned about wind farm noise"},
    {"sentiment": "positive", "text": "Renewable energy is crucial!"},
]
SAMPLE_REPORT = "The environmental impact is minimal..."


# --- Tool Functions ---
def get_lat_long(tool_context: ToolContext, location: str) -> Dict:
    """Retrieves latitude and longitude coordinates for a given location.

    Args:
        location (str): The name of the location.

    Returns:
        Dict: A dictionary containing latitude and longitude.
    """
    return {"latitude": 34.0522, "longitude": -118.2437}  # Mock Los Angeles

def get_earth_engine_images(tool_context: ToolContext, latitude: float, longitude: float) -> Dict:
    """Retrieves satellite images from Google Earth Engine based on coordinates.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        Dict: A dictionary containing image URLs.
    """
    return {"image_urls": SAMPLE_IMAGE_URLS}

def analyze_satellite_images(tool_context: ToolContext, image_urls: List[str]) -> Dict:
    """Analyzes satellite images to assess viability for an onshore wind farm using gemini-2.0-flash-001.

    Args:
        image_urls (List[str]): List of URLs pointing to satellite images in Google Cloud Storage.

    Returns:
        Dict: A dictionary containing viability assessment and detailed analysis.
    """
    client = genai.Client(
        vertexai=True,
        project="platinum-banner-303105",  # Replace with your project ID
        location="us-central1",
    )

    analysis_results = []

    for image_url in image_urls:
        try:
            image1 = types.Part.from_uri(
                file_uri=image_url,
                mime_type="image/png",
            )
            text1 = types.Part.from_text(text="""Analyze this satellite image for its suitability for an onshore wind farm. Consider factors like terrain, vegetation, existing infrastructure, and potential environmental impact.""")

            model = "gemini-2.0-flash-001"
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        image1,
                        text1
                    ]
                )
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                max_output_tokens=8192,
                response_modalities=["TEXT"],
                safety_settings=[types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ), types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ), types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ), types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )],
            )

            response_text = ""
            for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
            ):
                response_text += chunk.text

            analysis_results.append(response_text)

        except Exception as e:
            analysis_results.append(f"Error analyzing image {image_url}: {e}")

    # Combine analysis results into a summary
    summary = "\n".join(analysis_results)

    # Example viability assessment (replace with more sophisticated logic)
    if "suitable" in summary.lower() or "terrain" in summary.lower():
        viability = "high"
    elif "some" in summary.lower() or "moderate" in summary.lower():
        viability = "moderate"
    else:
        viability = "low"

    return {"viability": viability, "analysis": summary}

def search_social_media(tool_context: ToolContext, location: str) -> Dict:
    """Searches social media for posts related to a location.

    Args:
        location (str): The name of the location.

    Returns:
        Dict: A dictionary containing social media posts.
    """
    return {"posts": SAMPLE_SOCIAL_MEDIA_POSTS}

def check_lawsuits(tool_context: ToolContext, location: str) -> Dict:
    """Checks for existing lawsuits related to a location.

    Args:
        location (str): The name of the location.

    Returns:
        Dict: A dictionary indicating if lawsuits were found.
    """
    return {"lawsuits_found": False}

def check_landownership(tool_context: ToolContext, latitude: float, longitude: float) -> Dict:
    """Checks land ownership and acquisition difficulty based on coordinates.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        Dict: A dictionary containing ownership information and acquisition difficulty.
    """
    return {"ownership": "private", "acquisition_difficulty": "high"}

def analyze_environmental_report(tool_context: ToolContext, report: str) -> Dict:
    """Analyzes an environmental report for key findings.

    Args:
        report (str): The environmental report text.

    Returns:
        Dict: A dictionary containing key findings from the report.
    """
    return {"key_findings": ["No endangered species", "Low water usage"]}

def find_nearest_electrical_hub(tool_context: ToolContext, latitude: float, longitude: float) -> Dict:
    """Finds the nearest electrical hub and estimates connection costs.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        Dict: A dictionary containing distance and estimated connection cost.
    """
    return {"distance": 5, "estimated_cost": 50000}


# --- Agents ---
imagery_agent = Agent(
    name="satellite_imagery_agent",
    model="gemini-1.5-flash-002",
    tools=[get_lat_long, get_earth_engine_images, analyze_satellite_images],
    instruction="Use the tools at your disposal to Get lat/long, retrieve images, analyze, and assess viability.",
)

sentiment_agent = Agent(
    name="community_sentiment_agent",
    model="gemini-1.5-flash-002",
    tools=[search_social_media, check_lawsuits],
    instruction="Use the tools at your disposal to Analyze social media and legal data for community sentiment.",
)

land_agent = Agent(
    name="land_ownership_agent",
    model="gemini-1.5-flash-002",
    tools=[check_landownership],
    instruction="Use the tools at your disposal to Check land ownership and assess acquisition difficulty.",
)

impact_agent = Agent(
    name="environmental_impact_agent",
    model="gemini-1.5-flash-002",
    tools=[analyze_environmental_report],
    instruction="Use the tools at your disposal to Analyze the provided report.",
)

grid_agent = Agent(
    name="electrical_grid_agent",
    model="gemini-1.5-flash-002",
    tools=[find_nearest_electrical_hub],
    instruction="Use the tools at your disposal to Find nearest hub and estimate connection costs.",
)

control_agent = Agent(
    name="control_agent",
    model="gemini-1.5-flash-002",
    tools=[],
    flow='sequential',
    children=[imagery_agent, sentiment_agent, land_agent, impact_agent, grid_agent],
    instruction="""You are the control agent. Use the Agents at your disposal to gather the following info: 
    1) Analyze the satellite imagery of the location you are requesting - use the imagery_agent to gather that info
    2) Search for the social media and lawsuits of the location you are requesting to understand local sentiment - use the sentiment_agent to gather that info
    3) Check on landownership and understand if there are issues to be concerned with  - use the land_agent to gather that info
    4) Analyze the environmental impact reports and summarize the key info - use the impact_agent to gather that info
    5) get the information related to the electrical grid and any costs related to connecting to the nearest substation use the grid_agent to gather that info
    6) Once all the information is collected, combine outputs, and produce two separate reports (one per location).  
    7)Create a final report comparing both locations and providing a recommendation.
    - please use only the agents to fulfill all user request
""",
)

root_agent = control_agent
