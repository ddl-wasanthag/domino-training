library(httr2)

# Read environment variables
domino_api_url <- "https://prod-field.cs.domino.tech/v4/jobs/project"
domino_project_id <- Sys.getenv("DOMINO_PROJECT_ID")
domino_run_id <- Sys.getenv("DOMINO_RUN_ID")
domino_api_key <- Sys.getenv("DOMINO_USER_API_KEY")

# Construct the API URL
url <- paste0(domino_api_url, "/", domino_project_id, "/codeInfo/", domino_run_id)

# Make the API request
response <- request(url) |>
  req_headers("X-Domino-Api-Key" = domino_api_key) |>
  req_perform()

# Print response
resp_body_string(response)

