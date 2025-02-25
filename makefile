MODULENAME = fire_behavior 

help:
	@echo ""
	@echo "Welcome to my project!!!"
	@echo "To get started create an environment using:"
	@echo "	make init"
	@echo "	conda activate ./envs"
	@echo ""
	@echo "To generate project documentation use:"
	@echo "	make doc"
	@echo ""
	@echo "To Lint the project use:"
	@echo "	make lint"
	@echo ""
	@echo "To run unit tests use:"
	@echo "	make test"
	@echo ""
	

init:
	conda env create --prefix ./envs --file environment.yml

doc:
	pdoc3 --force --html --output-dir ./docs $(MODULENAME)

lint:
	pylint $(MODULENAME)

test:
	pytest -v $(MODULENAME)

.PHONY: init doc lint test 

