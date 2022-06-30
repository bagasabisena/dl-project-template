.PHONY: requirements clean lint upload_data download_data upload_outputs download_outputs environment jupyter

#################################################################################
# GLOBALS                                                                       #
#################################################################################

USERNAME := $(shell whoami)
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = {{ cookiecutter.data_bucket }}/projects/${USERNAME}/{{ cookiecutter.repo_name }}
PROJECT_NAME = {{ cookiecutter.repo_name }}
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
ifeq (True,$(HAS_CONDA))
	pip install -U pip setuptools wheel
	pip install -r requirements.txt
else
	@echo "Currently we only support conda-based project"
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

# Upload outputs to storage bucket (S3 or GCS)
upload_outputs:
{%- if cookiecutter.object_storage == 's3' %}
	aws s3 sync outputs/ s3://$(BUCKET)/outputs/
{%- elif cookiecutter.object_storage == 'gcs' %}
	gsutil -m rsync -r outputs/ gs://${BUCKET}/outputs/
{%- endif %}

## Download outputs from storage bucket (S3 or GCS)
download_outputs:
{%- if cookiecutter.object_storage == 's3' %}
	aws s3 sync s3://$(BUCKET)/outputs/ outputs/
{%- elif cookiecutter.object_storage == 'gcs' %}
	gsutil -m rsync -r gs://${BUCKET}/outputs/ outputs/
{%- endif %}

## Set up python interpreter environment
environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name $(PROJECT_NAME) python={{ cookiecutter.python_version }}
else
	@echo "Currently we only support conda-based project"
endif

## Run a jupyter lab server
jupyter:
	@echo "Starting a jupyterlab server. See the server address in the jupyter.log"
	./jupyter.sh;

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
