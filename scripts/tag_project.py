# python3 tag_project.py <project name to tag> <tag>
# 
# How to execute API external to Domino 
# TOKEN=`curl $DOMINO_API_PROXY/access-token`
# curl -X GET "https://<domino url>/v4/users?userName=<user name>" -H  "accept: application/json" -H  "Authorization: Bearer $TOKEN"

import os
import requests
import argparse
import logging

# Setup logging to print to standard output
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Parse arguments
parser = argparse.ArgumentParser(description="Tag a Domino project")
parser.add_argument("project_name", help="Name of the project to tag")
parser.add_argument("tag_name", help="Name of the tag to add to the project")
args = parser.parse_args()

# Get environment variables
domino_api_proxy = os.getenv('DOMINO_API_PROXY')
api_key = os.getenv('DOMINO_USER_API_KEY')

if not domino_api_proxy or not api_key:
    logging.error("Environment variables DOMINO_API_PROXY or DOMINO_USER_API_KEY are missing.")
    exit(1)

# Headers for API calls
headers = {
    'X-Domino-Api-Key': api_key,
    'Content-Type': 'application/json',
    'accept': 'application/json'
}

# Function to get project ID by name
def get_project_id_by_name(project_name):
    response = requests.get(f"{domino_api_proxy}/v4/projects", headers=headers)
    response.raise_for_status()
    projects = response.json()
    
    for project in projects:
        if project['name'] == project_name:
            return project['id']
    return None

# Function to tag a project
def tag_project(project_id, tag_name):
    response = requests.post(
        f"{domino_api_proxy}/v4/projects/{project_id}/tags",
        headers=headers,
        json={"tagNames": [tag_name]}
    )
    response.raise_for_status()
    logging.info(f"Tag '{tag_name}' added to project with ID '{project_id}'.")

# Main logic
project_id = get_project_id_by_name(args.project_name)

if project_id:
    tag_project(project_id, args.tag_name)
    logging.info(f"Project '{args.project_name}' successfully tagged with '{args.tag_name}'.")
else:
    logging.error(f"Project '{args.project_name}' not found.")
    exit(1)
