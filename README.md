# WILDCATCH
## Overview

This project aims to help **Northwestern University Men's Soccer** prepare for their next match by streamlining the analysis of match reports, team and player statistics, and generating actionable insights. It leverages advanced data analysis and OpenAI's large language models (LLMs) to provide:

- **Insights on the opposing team's playing style, strengths, and weaknesses.**  
- **Strategies and suggestions to counter the opposing team.**  
- **Lineup recommendations** based on the opposing team's players and our players' capabilities.  

The system automates the process of analyzing PDFs and structured data, making the preparation process efficient and targeted.

---

## Project Structure

```
soccerHackthon/
│
├── match-reports/           # Contains detailed PDF reports of matches for analysis.
│
├── player-data/             # Includes statistics and performance data for players.
│
├── src/                     # Source code for data processing and analysis.
│   │
│   ├── generate_graph.py    # Script to create visualizations.
│   ├── generate_prompt.py   # Uses OpenAI LLM to generate insights and strategies.
│   ├── helper.py            # Utility functions for data preprocessing and analysis.
│   ├── image_cropping.py    # Crops and processes images from match reports.
│   ├── player_recommend.py  # Provides player recommendations based on the opposing team's strength and weakness.
│   └──pipeline.ipynb        # Jupyter Notebook for generating insights on the opposing team.
│   
│
├── team-data/               # Includes statistics related data for Big Ten and ACC teams.
│
├── requirements.txt         # List of Python dependencies needed to run the project.
│
└── README.txt               # Documentation for the project.
```

---

## How It Works

1. **Data Organization**:
   - All data folders (e.g., **match-reports**, **player-data**, and **team-data**) should be placed in the **main folder**.

2. **Pipeline Functionality**:
   - The pipeline analyzes the data for a **specific team** (selected by the user).
   - Generates detailed insights on:
     - Opposing team's playing style.
     - Strengths and weaknesses.
     - Strategic suggestions to counter their gameplay.
     - Optimal lineup recommendations for NU Soccer.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <[repository_url](https://github.com/SimingSiming/soccerHackthon)>
   cd soccerHackthon
   ```

2. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

   Use the pipeline notebook to analyze a specific team:
   ```bash
   jupyter notebook src/pipeline.ipynb
   ```

---

## Key Features

- **Automated PDF Analysis**: Quickly extracts and summarizes match reports.  
- **Targeted Insights**: Provides a detailed breakdown of an opposing team's gameplay.  
- **AI-Powered Strategies**: Suggests counter-strategies using OpenAI's LLM.  
- **Lineup Optimization**: Recommends player lineups to maximize performance.  
- **Data Visualization**: Generates heatmaps, team formulation, and other key visuals.  

---

This tool provides a competitive edge to NU Soccer by making match preparation more efficient, data-driven, and strategic.
