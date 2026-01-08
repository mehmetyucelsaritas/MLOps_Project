class PromptBuilder:
    @staticmethod
    def build_wildfire_action_plan_prompt(user_input: str, retrieved_chunks: list[dict]) -> str:
        # Detailed examples (few-shot prompting)
        example_response = (
            "Example \n"
            "1. Location: Residential Area (Sector A)\n"
            "   Action: Evacuate all residents and assist vulnerable individuals.\n"
            "   Required Resources: Fire trucks, evacuation buses, emergency staff.\n"
            "   Reasoning: Rapid fire spread toward homes; immediate evacuation prevents casualties.\n"
            "   Navigation Hint: Use Main Street to guide residents to the safe zone near the Town Hall.\n\n"
        )

        system_instructions = (
            "You are an expert wildfire emergency response planner.\n"
            "Consider carefully the provided **weather conditions, wind, date, and time** below when creating your action plan.\n\n"
            "Always factor in temperature, wind speed and direction, and weather conditions when suggesting actions.\n\n"
            "Your task is to create a **clear and highly detailed step-by-step action plan** for firefighters responding to the wildfire situation below.\n\n"
            "For each action step, include:\n"
            "- Location: Where the action happens (specific waypoint, landmark, road, or region).\n"
            "- Action: Specific task to perform.\n"
            "- Required Resources: Equipment, personnel, tools.\n"
            "- Reasoning: Why this action is important.\n"
            "- Navigation Hint: Provide very specific guidance including road names, GPS coordinates, or detailed directions (e.g., 'go east on Main St. for 500 meters, then turn left onto Oak Road').\n\n"
            "Make the navigation hints as concrete and actionable as possible.\n\n"
            "Do not invent fictional waypoints or codes unless such data is provided in the wildfire situation or retrieved documents.\n\n"
            "Examples of well-structured responses:\n\n"
            f"{example_response}\n"
            "Now, based on the retrieved documents and the current wildfire situation provided below, create a complete action plan covering all necessary locations and actions."
            "After the action plan, list the **sources and regions** used in your answer explicitly.\n\n"
            "Format your response as follows:\n"
            "### ACTION PLAN:\n"
            "...\n\n"
            "### SOURCES USED:\n"
            "- [Source] (Region: [Region])\n"
            "- [Source] (Region: [Region])\n"
        )

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            summary = chunk.get("summary", chunk["text"])  # fallback to text if no summary
            source_name = chunk.get("source_name", chunk.get("source", "Unknown Source"))
            region = chunk.get("region", "Unknown Region")

            part = f"{i}. {summary}\n(Source: {source_name}, Region: {region})"
            context_parts.append(part)

        context_text = "\n\n".join(context_parts)

        # Final prompt
        final_prompt = f"{system_instructions}\n" f"--- Retrieved Documents ---\n\n{context_text}\n\n" f"--- Wildfire Situation ---\n\n{user_input}\n\n" f"--- Your Action Plan ---"

        return final_prompt
