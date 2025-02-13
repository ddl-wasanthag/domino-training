#!/bin/bash
set -x
curl https://prod-field.cs.domino.tech/v4/jobs/project/$DOMINO_PROJECT_ID/codeInfo/$DOMINO_RUN_ID -H "X-Domino-Api-Key:  $DOMINO_USER_API_KEY"

