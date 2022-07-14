
.PHONY: venv
venv:
	 python3 -m venv venv
	# source venv/bin/activate

.PHONY: upgrade
upgrade:
	pip install --upgrade pip -vvv

.PHONY: freeze
freeze:
	pip freeze > requirements.txt

.PHONY: install
install:
	pip install -r requirements.txt


.PHONY: build
build:
	pip install .