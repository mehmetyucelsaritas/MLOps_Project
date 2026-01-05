
# FireGPT


## Getting started

We provide a dockerfile to build the app and get its dependencies with less headache.

First, install Docker for your platform.

Clone the repository:

```
git clone git@github.com:mehmetyucelsaritas/FireGPT.git
```
Even though the LLM models are automatically downloaded in the docker image, we recommend to download the models prior, as long recompilation caused by potential build issues can be avoided this way:

```
pip install huggingface-hub

hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models
hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ./models
hf download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir ./models
```

Next, execute this line in a terminal (PowerShell) in the repositories root:

```
docker build -t amifgptv1 . ; docker run -p 8501:8501 -it amifgptv1
```

In the terminal you should see the link to the web interface. 
Otherwise go to http://127.0.0.1:8501

For testing you can use following synthetic query (generated with ChatGPT, based on the Eaton Fire):  

### Querry
I’m facing a wildfire that started at coordinate (2.0,1.5) in the northwest quadrant of our operational map (grid origin at top-left (0,0), units in km). The fire has spread southeast, roughly following a ridge from (2.5,2.0) to (4.0,3.5), and it’s approaching the foothill community zone in the rectangular area between X =4.5–6.5km and Y=3.5–5.0km. There’s a natural drainage gulch along Y ≈1.0 where ember‑spotting is common, and a human-made access road/firebreak runs at X ≈3.0 from Y=0.5 to Y=5.0. Two evacuation shelters are located outside the fire area at approximately (8.5,6.5) and (9.0,7.0).

We have available assets:
• 1 Thermal IR drones   
• 1 Drone with water carrying ability   
• 2 trucks and crew

### example result with mistral-7b:

    ACTION PLAN:

        Location: Fire Command Center (approximately (0.0, 0.0)) Action: Assemble the incident management team and assess the current situation using available resources and weather data. Required Resources: Thermal IR drones, weather data, operational map, communication equipment. Reasoning: Early assessment and understanding of the fire situation is crucial for effective response planning and resource allocation. Navigation Hint: The fire command center is located at the grid origin of the operational map.

        Location: Fire Perimeter near (4.0, 3.5) Action: Deploy the first firefighting crew to contain the fire and secure the human-made firebreak. Required Resources: 1 truck and crew. Reasoning: The fire is approaching the foothill community zone, and securing the firebreak is essential to prevent the fire from spreading further. Navigation Hint: Travel south along the Y-axis from the fire command center to reach the fire perimeter.

        Location: Fire Perimeter near (2.5, 2.0) Action: Deploy the second firefighting crew to engage the fire from the northwest and create a containment line. Required Resources: 1 truck and crew. Reasoning: The fire is following a ridge from this location, and containing it here will help prevent the fire from spreading further in that direction. Navigation Hint: Travel west along the X-axis from the fire command center to reach the fire perimeter.

        Location: Natural Drainage Gulch (approximately Y ≈ 1.0) Action: Deploy the thermal IR drone to monitor for ember spotting and identify potential hotspots. Required Resources: Thermal IR drone. Reasoning: The natural drainage gulch is a common area for ember spotting, and early detection and containment of these hotspots is crucial to prevent the fire from spreading. Navigation Hint: Travel south along the Y-axis from the fire command center to reach the natural drainage gulch.

        Location: Evacuation Shelter at (8.5, 6.5) Action: Deploy the drone with water carrying ability to support evacuation efforts and provide water to residents. Required Resources: Drone with water carrying ability. Reasoning: Evacuation shelters need water and support to accommodate residents, and the drone can help provide this essential resource. Navigation Hint: Travel northwest from the fire command center to reach the evacuation shelter.

    SOURCES USED:

        Vegetation Fire Fighting Training Manual for Fire Departments Version 1, Region: Germany
        Fire Management Today Volume 63, No. 4, Fall 2003, Region: International/United States/Australia/New Zealand
        The South Canyon Fire Behavior Report, Region: Colorado, United States

### Queries for evaluation
You can find test_queries.json under ./testset for evaluation. For your reference you can also find our evaluation results for each query with different models in ./eval/eval_results. M7 stands for Mistral-7B; P3 stands for Phi3 Mini; and TL stands for TinyLlama.  